import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

img_bgr = cv2.imread(R'C:\Users\13191\Desktop\111.jpg', cv2.IMREAD_COLOR)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
cv2.imwrite(r'C:\Users\13191\Desktop\222.jpg', img_lab)
