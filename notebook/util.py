import json
import os
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np

test_images = sorted(Path('data/test_images').glob('**/*.png'))
train_images = sorted(Path('data/train_images').glob('**/*.png'))
train_annotations = sorted(
    Path('data/train_annotations').glob('**/*.json'))

rect_keys = ['引戸', '折戸', '開戸']
polygon_keys = ['LDK', '廊下', '浴室']


def read_json(json_path):
    json_load = open(json_path, 'r')
    json_file = json.load(json_load)
    return json_file


def draw_rectangle(img, pts):
    """"矩形を描写する

    Args:
        img (ndarray): cv2.imreadで読み込んだ画像
        pts (list[int]):　座標

    Returns:
        [type]: imgに矩形を描写した画像
    """
    return cv2.rectangle(img, tuple(pts[:2]), tuple(pts[2:]), (255, 0, 0), 2)


def draw_polylines(img: np.ndarray, pts: np.ndarray, rgb=(0, 0, 255)):
    """画像情に多角形を描写する

    Args:
        img (np.ndarray): cv2.imreadで読み込んだ画像
        pts (list(int)): 座標

    Returns:
        [np.ndarray]: imgに多角形を描写した画像
    """
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], True, rgb, 2)


def save_imgs_with_figure(num: int):
    """"画像にアノテーションし（矩形や多角形を描写し）、保存する

    Args:
        num (int): 画像やjsonの番号
    """
    img = cv2.imread(train_images[num])
    load_json = read_json(str(train_annotations[num]))
    for item in load_json['labels']:
        # doors
        if item in rect_keys:
            for pts in load_json['labels'][item]:
                pts = np.array(pts, np.int)
                draw_rectangle(img, pts)
        # rooms
        elif item in polygon_keys:
            for pts in load_json['labels'][item]:
                pts = np.array(pts, np.int32)
                draw_polylines(img, pts)
        else:
            for pts in load_json['labels'][item]:
                pts = np.array(pts, np.int32)
                draw_polylines(img, pts, (0, 255, 0))
    cv2.imwrite('/content/drive/MyDrive/Colab_Notebooks/kaggle/panasonic/'
                'data/imgs_anno/' +
                os.path.basename(train_images[num]), img)


def convert_to_coco_format(train_images_list_length: int):
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "door", "id": 1, "name": "引戸"},
        {"supercategory": "door", "id": 2, "name": "折戸"},
        {"supercategory": "door", "id": 3, "name": "開戸"},
        {"supercategory": "room", "id": 4, "name": "LDK"},
        {"supercategory": "room", "id": 5, "name": "廊下"},
        {"supercategory": "room", "id": 6, "name": "浴室"},
        {"supercategory": "room", "id": 7, "name": "洋室"},
    ]
    images = list()
    annotations = list()
    image_id = 0

    for num in range(train_images_list_length):
        # json, image
        json_file = read_json(train_annotations[num])
        img_path = train_images[num]
        img = cv2.imread(str(img_path))

        image_id += 1
        image = dict()
        image['file_name'] = os.path.basename(img_path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = image_id
        print("File Name: {} and image_id {}".format(img_path, image_id))
        images.append(image)
        id1 = 1

        for value in attrDict["categories"]:
            print('value: ', value["name"])
            annotation = dict()
            if str(obj['name']) == value["name"]:
                annotation["iscrowd"] = 0
                annotation["image_id"] = image_id
                x1 = int(obj["bndbox"]["xmin"]) - 1
                y1 = int(obj["bndbox"]["ymin"]) - 1
                x2 = int(obj["bndbox"]["xmax"]) - x1
                y2 = int(obj["bndbox"]["ymax"]) - y1
                annotation["bbox"] = [x1, y1, x2, y2]
                annotation["area"] = float(x2 * y2)
                annotation["category_id"] = value["id"]
                annotation["ignore"] = 0
                annotation["id"] = id1
                annotation["segmentation"] = [
                    [x1, y1, x1, (y1 + y2), (x1 + x2), (y1 + y2),
                        (x1 + x2), y1]]
                id1 += 1
                annotations.append(annotation)

            else:
                print("File: {} doesn't have any object".format(file))

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    jsonString = json.dumps(attrDict)
    with open("train.json", "w") as f:
        f.write(jsonString)


if __name__ == '__main__':
    num = len(train_images)
    convert_to_coco_format(3)
    # import multiprocessing as multi
    # from multiprocessing import Pool

    # nums = len(train_images)
    # # mulitiprocessing
    # core = multi.cpu_count()
    # print(core)
    # p = Pool(core)
    # imap = p.imap(save_imgs_with_figure, list(range(nums)))
    # result = list(tqdm(imap, total=nums))
