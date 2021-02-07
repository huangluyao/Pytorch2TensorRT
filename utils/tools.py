import numpy as np
import math
import cv2, random
from pycocotools.coco import COCO
from tqdm import tqdm
import os


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def plot_boxes_cv2(img, boxes, class_names=None, color=None, savename=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def json_to_txt(ann_path, img_dir, yolo_txt,xyxy=False):
    def cati2name(coco):
        classes = dict()
        for cat in coco.dataset['categories']:
            classes[cat['id']] = cat['name']
        return classes


    def get_img_info(img, coco, classes):
        file_name = img['file_name']
        file_path = os.path.join(img_dir, file_name)
        I = cv2.imread(file_path)
        ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        json_path = file_path.replace(".jpg", ".json")
        img_info = {}
        img_info['img'] = file_name
        anns_new = []
        for ann in anns:
            ann['name'] = classes[ann['category_id']]
        img_info['anns'] = anns

        return img_info

    coco = COCO(ann_path)
    classes = cati2name(coco)
    img_ids = coco.getImgIds()

    with open(yolo_txt, 'w') as fp:
        for img_id in tqdm(img_ids):
            img = coco.loadImgs(img_id)[0]
            img_info = get_img_info(img, coco, classes)
            image_path = os.path.join(img_dir, img_info["img"])
            fp.write(image_path + " ")
            for anns in img_info["anns"]:
                bbox =[float(b) for b in anns["bbox"][:4]]
                id = anns["category_id"]

                if xyxy:
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]

                sbbox = ""
                for point in bbox:
                    sbbox += str(point) + ","
                fp.write("%s%s " %(sbbox, id))
            fp.write("\n")


def calculate_means_and_std(train_txt_path="./dataset/Bead_neg_dataset/train.txt"):

    img_h, img_w = 416, 416
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []
    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)   # shuffle , 随机挑选图片

    for i in range(len(lines)):
            img_path = "../"+lines[i].split(" ")[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_h, img_w))
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print(i)

    imgs = imgs.astype(np.float32)/255.
    for i in range(3):
        pixels = imgs[:,:,i,:].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
    means.reverse() # BGR --> RGB
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

if __name__ == "__main__":
    # train_txt_path = "../dataset/Bead_neg_dataset/train.txt"
    # calculate_means_and_std(train_txt_path)

    json_to_txt()