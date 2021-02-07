import os, random
import cv2, json, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from augmentation.pipelines.compose import Compose


class YOLO_Dataset(Dataset):
    def __init__(self, size,label_path, augmentation_path, mode="train"):
        super(YOLO_Dataset,self).__init__()
        self.image_size = size
        self.img_info = self.get_img_info(label_path)
        self.imgs = list(self.img_info.keys())
        self.number_boxes = 40
        with open(augmentation_path) as fp:
            data = json.load(fp)
            config = data['data']
        self.mode = mode
        if mode == 'train':
            self.train_pipeline = Compose(config['train'], bbox_params= {"format":"coco"})
        else:
            self.train_pipeline = Compose(config['test'], bbox_params= {"format":"coco"})

    def get_img_info(self, lable_path):
        img_info = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            img_info[data[0]] = []
            for i in data[1:]:
                if i != '\n':
                    img_info[data[0]].append([int(float(j)) for j in i.split(',')])
        f.close()
        return img_info

    def __getitem__(self, item):
        if self.mode != "train":
            return self._get_val_item(item)
        img_path = self.imgs[item]
        bboxes = np.array(self.img_info.get(img_path), dtype=np.float)
        img = cv2.imread(img_path)

        image = visualize_bbox_and_id(img, bboxes, **{1:"balloon"})
        cv2.imshow("result", image)
        cv2.waitKey()
        img, bboxes = self.img_transform(img, bboxes)

        out_bboxes1 = np.zeros([self.number_boxes, 5])
        if len(bboxes)> 0:
            out_bboxes1[:min(len(bboxes), self.number_boxes)] = bboxes[:min(len(bboxes), self.number_boxes)]
            out_bboxes1[...,2:4] += out_bboxes1[...,0:2]
        out_bboxes1 = torch.from_numpy(out_bboxes1)
        return img, out_bboxes1

    def img_transform(self, img, bboxes):
        data = {"type": "object_detection"}
        data["image"] = img
        data["bboxes"] = bboxes
        augment_result = self.train_pipeline(data)
        image = augment_result["image"]
        bboxes = augment_result["bboxes"]
        # 转成tensor格式
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        # bboxes = torch.tensor(bboxes)
        return image, bboxes


    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.img_info.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data = {"type": "object_detection"}
        data["image"] = img
        data["bboxes"] = bboxes_with_cls_id
        augment_result = self.train_pipeline(data)
        img = augment_result["image"]
        bboxes_with_cls_id = augment_result["bboxes"]


        bboxes_with_cls_id =np.array([[int(b) for b in bbox_with_cls_id] for bbox_with_cls_id in bboxes_with_cls_id])

        bboxes_with_cls_id[...,2:4] += bboxes_with_cls_id[...,:2]
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        image_id = int(os.path.splitext(img_path.split("/")[-1])[0])
        target['image_id'] = torch.tensor(image_id)
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    from utils.vis import visualize_bbox_and_id
    # json_to_txt()

    category_id_to_name = {"bead":1}
    label_path = "../dataset/Bead_neg_dataset/train.txt"
    aug_path = "../data/augmentation.json"
    db = YOLO_Dataset(label_path, aug_path)
    data = iter(db)
    for image, bbox in data:
        cv2.imshow("src", image)
        dst = visualize_bbox_and_id(image,bbox,{1:"bead"})

        cv2.imshow("dst", dst)
        cv2.waitKey()
        pass

