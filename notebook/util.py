import json

import cv2


def read_json(json_path: str):
    json_load = open(json_path)
    json_file = json.load(json_load)
    return json_file


def draw_rectangle(img, pts):
    return cv2.rectangle(img, tuple(pts[:2]), tuple(pts[2:]), (255, 0, 0), 2)


def draw_polylines(img, pts):
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], True, (0, 0, 255), 2)
