import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def normalize_coefficients(coeff):
    """
    将小波系数的值范围缩放到 0-255
    """
    coeff_min = np.min(coeff)
    coeff_max = np.max(coeff)
    coeff_normalized = 255 * (coeff - coeff_min) / (coeff_max - coeff_min)
    return coeff_normalized.astype(np.uint8)

def pad_to_square(image):
    """
    将输入图像填充成正方形
    """
    height, width = image.shape[:2]
    if height == width:
        return image
    
    size = max(height, width)
    padded_image = np.zeros((size, size), dtype=image.dtype)
    
    if height > width:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        padded_image[:, pad_left:-pad_right] = image
    else:
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
        padded_image[pad_top:-pad_bottom, :] = image
    
    return padded_image

def recursive_layout(cA, details):
    """
    递归地布局小波系数
    """
    if not details:
        return normalize_coefficients(cA)
    
    cH, cV, cD = details[0]
    next_cA, next_details = cA, details[1:]
    
    # 递归处理下一层
    top_left = recursive_layout(next_cA, next_details)
    top_right = normalize_coefficients(cH)
    bottom_left = normalize_coefficients(cV)
    bottom_right = normalize_coefficients(cD)
    
    # 调整所有子图像的大小为相同
    size = top_left.shape[0] + top_right.shape[0]
    top_left = pad_to_square(top_left)
    top_right = pad_to_square(top_right)
    bottom_left = pad_to_square(bottom_left)
    bottom_right = pad_to_square(bottom_right)
    
    top = np.concatenate((top_left, top_right), axis=1)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    full_image = np.concatenate((top, bottom), axis=0)
    
    return full_image

# 示例用法
image_path = 'data/test/87197679-1721717643835.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 将图像填充成正方形
square_image = pad_to_square(image)

# 进行小波变换
wavelet = 'db1'
level = 2
coeffs = pywt.wavedec2(square_image, wavelet, level=level)

# 将小波变换结果递归拼接成一副大图
full_image = recursive_layout(coeffs[0], coeffs[1:])

# 保存和显示拼接后的图像
output_path = 'wavelet_coefficients.png'
cv2.imwrite(output_path, full_image)

# 显示结果
plt.figure(figsize=(10, 10))
plt.title('Wavelet Coefficients')
plt.imshow(full_image, cmap='gray')
plt.axis('off')
plt.show()
