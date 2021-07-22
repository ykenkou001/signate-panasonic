import json
import os
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from tqdm import tqdm

cwd = os.getcwd()
print('cwd: ', cwd)
train_images = sorted(Path(cwd + '/Yet-Another-EfficientDet-Pytorch/datasets/'
                          'detection_door_room/train').glob('**/*.jpg'))
val_images = sorted(Path(cwd + '/Yet-Another-EfficientDet-Pytorch/datasets/'
                        'detection_door_room/val').glob('**/*.jpg'))
# train_images = train_images + val_images

train_annotations = sorted(
    Path(cwd + '/data/train_annotations').glob('**/*.json'))

def read_json(json_path: str) -> dict:
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


def convert_to_coco_format(filename_list: list, save_path: str) -> None:
    # coco format json
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
    images = []
    annotations = []
    image_id = 0

    train_json_path = ''

    for file_name in tqdm(filename_list):
        js = read_json(str(train_annotations[file_name]))
        img = cv2.imread(str(train_images[file_name]))

        # imageの情報
        image_id += 1
        image = {}
        image['file_name'] = train_images[num].name
        image['height'], image['width'] = img.shape[0], img.shape[1]
        image['id'] = image_id
        images.append(image)

        id1 = 1
        for value in attrDict["categories"]:
            if value['name'] in list(js['labels'].keys()):
                for elem in js['labels'][value['name']]:
                    anno_dict = {}
                    elem = np.array(elem).flatten()
                    anno_dict['iscrowd'] = 0
                    anno_dict['image_id'] = image_id
                    xmin, xmax = min(elem[::2]), max(elem[::2])
                    ymin, ymax = min(elem[1::2]), max(elem[1::2])
                    anno_dict['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
                    anno_dict['area'] = (xmax - xmin) * (ymax - ymin)
                    anno_dict['category_id'] = value['id']
                    anno_dict['id'] = id1
                    anno_dict['segmentation'] = elem.tolist()
                    id1 += 1
                    annotations.append(anno_dict)

    attrDict["images"] = images
    attrDict["annotations"] = annotations

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(attrDict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    pass
    # nums = len(train_images)
    # # mulitiprocessing
    # core = multi.cpu_count()
    # print(core)
    # p = Pool(core)
    # imap = p.imap(save_imgs_with_figure, list(range(nums)))
    # result = list(tqdm(imap, total=nums))
