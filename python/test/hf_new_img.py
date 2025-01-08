import os
import json
import numpy as np
import cv2


dx = 7.765754
dy = 7.78159
error = 0.12

dx_offset = 100
dy_offset = 100


def new_img(w, h, pts):
    image = np.zeros((w, h, 3), dtype=np.uint8)
    r = error * 1000 / dx
    for item in pts:
        cv2.circle(image, (item[0], item[1]), int(r), (255, 255, 255), -1)
    cv2.imwrite('./dst.jpg', image)


x_coords1 = [2.51, 6.01, 9.45, 12.95, 16.45, 19.95, 23.45, 26.95,
             30.45, 33.95, 37.45, 40.95, 44.45, 47.95, 51.45, 54.95, 58.45]
y_coords1 = [1.8, 5.4]

x_coords2 = [0.0, 1.5, 3.5, 5.0, 7.0, 8.5, 10.5, 12.0, 14.0, 15.5, 17.5, 19.0, 21.0, 22.5, 24.5, 26.0, 28.0, 29.5,
             31.5, 33.0, 35.0, 36.5, 38.5, 40.0, 42.0, 43.5, 45.5, 47.0, 49.0, 50.5, 52.5, 54.0, 56.0, 57.5, 59.5, 61.0]
y_coords2 = [0, 3.6, 7.2]


def generate_pts():
    pts = []
    for item_x in x_coords1:
        for item_y in y_coords1:
            pts.append([int(item_x * 1000 / dx + dx_offset),
                       int(item_y * 1000 / dy + dy_offset)])

    for item_x in x_coords2:
        for item_y in y_coords2:
            pts.append([int(item_x * 1000 / dx+dx_offset),
                       int(item_y * 1000 / dy+dy_offset)])

    new_img(7000, 9344, pts)


generate_pts()
