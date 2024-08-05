import copy
import os
import cv2
import numpy as np
import glob
from image_contour import draw, get_contour


def draw_hist(data, save_path):
    #画出模糊度的分布直方图，用来确定模糊阈值
    import matplotlib.pyplot as plt
    plt.hist(data, bins=100, range=[0, 1000], alpha=0.2)  # 设置分组数为10，可以根据需要调整
    plt.title('直方图')  # 设置图表标题
    plt.xlabel('数据值')  # 设置X轴标签
    plt.ylabel('频数')  # 设置Y轴标签
    # 显示直方图
    plt.savefig(save_path)


def get_blur_mask(img, patch_size=9, blur_threshold=50, hit_threshold=5,connected_area_threshold=1000):
    h, w = img.shape[:2]
    half_patch_size = patch_size // 2
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    #镜像padding边界
    bordered_laplacian = cv2.copyMakeBorder(laplacian, half_patch_size, half_patch_size, half_patch_size,
                                            half_patch_size, cv2.BORDER_REFLECT)
    accumulate_map = np.zeros_like(laplacian)
    var_list = []
    for h_middle_idx in range(h):
        for w_middle_idx in range(w):
            patch = bordered_laplacian[h_middle_idx:h_middle_idx + patch_size, w_middle_idx:w_middle_idx + patch_size]
            patch_var = patch.var()
            var_list.append(patch_var)
            if patch_var < blur_threshold:#小于该阈值，则认为该方块小图是模糊的，accumulate_map中对应的元素加1
                accumulate_map[h_middle_idx - half_patch_size:h_middle_idx + patch_size + 1,
                w_middle_idx - half_patch_size:w_middle_idx + half_patch_size + 1] += 1
    draw_hist(var_list, "blur_hist.png")
    mask = np.where(accumulate_map > hit_threshold, 255, 0)#命中次数大于命中阈值时，则认为该点属于模型区域。255白色是模糊区域，0黑色是清晰区域
    mask=mask.astype(np.uint8)
    # 创建腐蚀和膨胀的内核
    kernel = np.ones((2, 2), np.uint8)
    # 膨胀操作
    mask = cv2.dilate(mask, kernel, iterations=10)
    # 腐蚀操作
    # mask = cv2.erode(mask, kernel, iterations=1)

    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    # 打印连通区域数量
    print("连通区域数量:", num_labels - 1)

    # 打印每个部分的面积
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        print(f"连通区域 {label} 的面积: {area}")
        if area<connected_area_threshold:
            #面积小于阈值的区域不认为是模糊区域
            print(area,label)
            mask=np.where(labels==label,0,mask)
    return mask


if __name__ == "__main__":
    img_path_list = glob.glob("../test2/*.jpg")
    output_dir = "../test2_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for img_path in img_path_list:
        name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, name)
        print(name)
        img = cv2.imread(img_path)
        #获取模糊的区域mask
        mask = get_blur_mask(img)
        cv2.imwrite(os.path.join(output_dir,"mask_"+name), mask)
        #得到模糊区域的边界
        contours = get_contour(mask)
        #画出边界
        draw(img, contours, True, False, save_path)
