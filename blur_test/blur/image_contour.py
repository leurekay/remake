import copy
import os
import cv2
import numpy as np
import glob

def get_contour(img, threshold=128,use_morphology=True):
    if len(img.shape)==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img=copy.deepcopy(img)
    mask=np.where(gray_img<threshold,255,0).astype(np.uint8)
    if use_morphology:
        # 创建腐蚀和膨胀的内核
        kernel = np.ones((2, 2), np.uint8)
        # 膨胀操作把缺失的边界连上
        mask = cv2.dilate(mask, kernel, iterations=10)
        # 使用腐蚀操作
        # mask = cv2.erode(mask, kernel, iterations=1)
    #3.轮廓提取
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw(img, contours, is_draw_edge=True, is_fill=True, save_path=None):
    img_new = copy.deepcopy(img)
    if is_draw_edge:
        img_new = cv2.drawContours(img_new, contours, -1, (0, 0, 0),2)
    if is_fill:
        cv2.drawContours(img_new, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
    if save_path:
        cv2.imwrite(save_path, img_new)


if __name__ == "__main__":
    img_path_list = glob.glob("../test1/*.png")
    output_dir = "../test1_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for img_path in img_path_list:
        name = os.path.basename(img_path)
        print(name)
        img = cv2.imread(img_path)
        contours = get_contour(img)
        draw(img, contours, True, True, os.path.join(output_dir, name))
