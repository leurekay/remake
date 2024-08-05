import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def detect_moire(image, wavelet='db1', level=2, threshold_factor=1.5):
    """
    检测图像中的摩尔纹。
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行小波变换
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    cA, *details = coeffs
    
    # 检测摩尔纹
    
    for detail_level in details:
        cH, cV, cD = detail_level
        moire_mask = np.zeros_like(cD, dtype=bool)
        threshold = threshold_factor * np.mean(cD) + np.std(cD)
        moire_mask = moire_mask | (np.abs(cD) > threshold)
        # 显示结果
        # plt.figure(figsize=(12, 6))
        # plt.title('Diagonal Detail Coefficients (cD1)')
        # plt.imshow(cD, cmap='gray')
        # plt.show()
    
    return np.any(moire_mask)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    检测图像中的边缘。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def is_rephotographed(image, moire_threshold_factor=13.5, edge_threshold=1000000000):
    """
    判断图片是否翻拍。
    """
    # 检测摩尔纹
    moire_detected = detect_moire(image, threshold_factor=moire_threshold_factor)
    
    # 检测边缘特征
    edges = detect_edges(image)
    edge_count = np.sum(edges > 0)
    
    # 判断条件
    if moire_detected or edge_count > edge_threshold:
        return True
    return False

if __name__=="__main__":

    # 示例用法
    image_path = 'data/test/87197679-1721717643835.jpeg'
    image = cv2.imread(image_path)

    # 判断图片是否翻拍
    rephotographed = is_rephotographed(image)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    edges = detect_edges(image)
    plt.subplot(1, 2, 2)
    plt.title('Detected Edges')
    plt.imshow(edges, cmap='gray')
    plt.show()
    print(f"Is the image rephotographed? {'Yes' if rephotographed else 'No'}")




    # import glob
    # path_list=glob.glob("data/image_2000/*/*.jpg")
    # print(path_list[:10])

    # for i,image_path in enumerate(path_list):
    #     if i%100==0:
    #         print(i)
    #     image = cv2.imread(image_path)
    #     is_remake=is_rephotographed(image)
    #     if is_remake:
    #         print("remake path : ",image_path)
