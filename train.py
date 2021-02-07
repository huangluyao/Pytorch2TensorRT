import argparse
import json, os
import torch
from models.yolov4 import Yolov4
from utils.log import get_logger
from utils.yolo_dataset import YOLO_Dataset
from utils.loss import Yolo_Loss
from torch.utils.data import DataLoader
from utils.evaluate import evaluate
import torch.nn as nn
from tqdm import tqdm
from utils.tools import json_to_txt

# ann_path = "./dataset/balloon_det/val.json"
# img_dir = "/home/hly/workspace/code/py_code/yolo/dataset/balloon_det/val/normal/images"
# yolo_txt = "./dataset/balloon_det/val.txt"
#
# json_to_txt(ann_path, img_dir, yolo_txt, xyxy=True)

def collate_fn(batch):
    return tuple(zip(*batch))

def run(cfg, logger):
    # 1. Get Model
    logger.info(f'Conf | use model_name yolov4')
    logger.info(f'Conf | number of classes {cfg["n_classes"]}')
    model = Yolov4(cfg["pretrained"], n_classes=cfg["n_classes"])
    eval_model = Yolov4(cfg["pretrained"], n_classes=cfg["n_classes"], inference=True)
    eval_model.to(cfg["device"])
    # 2. Whether to use multi-gpu training
    gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        eval_model = nn.DataParallel(eval_model, device_ids=gpu_ids)
    model = model.to(cfg["device"])
    if "weight" in cfg.keys():
        logger.info(f'Conf | use weight {cfg["weight"]}')
        model.load_state_dict(torch.load(cfg["weight"]))

    # 3. Get Yolo Dataset
    logger.info(f'Conf | use train_label_path {cfg["train_label_path"]}')
    logger.info(f'Conf | use test_label_path {cfg["test_label_path"]}')
    logger.info(f'Conf | use augmentation_path {cfg["augmentation_path"]}')
    train_dataset = YOLO_Dataset(cfg["image_h"],cfg["train_label_path"], augmentation_path=cfg["augmentation_path"], mode="train")
    test_dataset = YOLO_Dataset(cfg["image_h"],cfg["test_label_path"], augmentation_path=cfg["augmentation_path"], mode="test")
    n_train = len(train_dataset)
    n_val = len(test_dataset)
    logger.info(f'Conf | use num_workers {cfg["num_workers"]}')
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn,
                             num_workers=cfg["num_workers"],pin_memory=True)

    # 4. Get optimizer and loss
    logger.info(f'Conf | use optimizer Adam')
    logger.info(f'Conf | use lr {cfg["lr"]}')
    logger.info(f'Conf | use milestones {cfg["milestones"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9,0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg["milestones"])
    criterion = Yolo_Loss(device=cfg["device"], batch=cfg["batch_size"] , n_classes=cfg["n_classes"], anchors=cfg["anchors"])

    # 5. model train
    logger.info(f'Conf | use epochs {cfg["epochs"]}')
    epochs = cfg["epochs"]
    best_AP50 = 0.
    best_epoch = -1
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs} loss={epoch_loss}') as pbar:
            # evaluator = evaluate(eval_model, test_loader, cfg, cfg["device"])

            model.train()
            for i, data in enumerate(train_loader):
                images = data[0].to(device=cfg["device"], dtype=torch.float32)
                bboxes = data[1].to(device=cfg["device"])

                outs = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(outs, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()
                optimizer.step()

                epoch_loss = (epoch_loss + loss.item()) / (i+1)

                pbar.set_postfix(**{"Epoch":epoch,
                                    'loss (batch)': loss.item(),
                                    'loss_xy': loss_xy.item(),
                                    'loss_wh': loss_wh.item(),
                                    'loss_obj': loss_obj.item(),
                                    'loss_cls': loss_cls.item(),
                                    'loss_l2': loss_l2.item(),
                                    'lr': scheduler.get_lr()[0] * cfg["batch_size"]
                                    })

                pbar.update(images.shape[0])
        scheduler.step()

        eval_model.load_state_dict(model.state_dict())
        evaluator = evaluate(eval_model,test_loader,cfg, cfg["device"])
        stats = evaluator.coco_eval['bbox'].stats
        logger.info(f'Test | Val AP {stats[0]}')
        logger.info(f'Test | Val AP50 {stats[1]}')
        logger.info(f'Test | Val AP75 {stats[2]}')
        logger.info(f'Test | Val AP_small {stats[2]}')
        logger.info(f'Test | Val AP_medium {stats[2]}')
        logger.info(f'Test | Val AP_large {stats[2]}')
        logger.info(f'Test | Val AR1 {stats[2]}')
        logger.info(f'Test | Val AR10 {stats[2]}')
        logger.info(f'Test | Val AR100 {stats[2]}')
        logger.info(f'Test | Val AR_small {stats[2]}')
        logger.info(f'Test | Val AR_medium {stats[2]}')
        logger.info(f'Test | Val AR_large {stats[2]}')


        if best_AP50 <= stats[1]:
            best_AP50 = stats[1]
            best_epoch = epoch
            save_path = os.path.join(cfg['logdir'], 'best_train_miou.pth')
            if len(gpu_ids) > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

    logger.info(f'END | best MIou in Test is  {best_AP50:.5f}')
    logger.info(f'Save | [{best_epoch + 1:3d}/{cfg["epoch"]}] save the best model to {save_path}')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default="data/Dead_YOLO.json",
                        help="Configuration to use")

    args = parser.parse_args()
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logger = get_logger(cfg, args.config)

    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    run(cfg, logger)

