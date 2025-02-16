import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_pip_black(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # colors taken from color picker online , and threshold decided randomly using gpt (could be altered for a better output)
    lower_green = np.array([80, 150, 80])  
    upper_green = np.array([90, 255, 255])  
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)  
    image_black = image.copy()
    image_black[mask > 0] = [0, 0, 0] # setting the area to black color
    return image_black, mask


def image_agcwd_color(img, a=0.25, truncated_cdf=False):
    
    img = cv2.resize(img, (512, 512))
    #splitting into channels
    channels = cv2.split(img)
    new_channels = []
    for channel in channels:
        h, w = channel.shape[:2]
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        prob_normalized = hist / hist.sum()
        unique_intensity = np.unique(channel)
        prob_min = prob_normalized.min()
        prob_max = prob_normalized.max()
        pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
        pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
        pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
        prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd) if truncated_cdf else 1 - cdf_prob_normalized_wd
        channel_new = channel.copy()
        for i in unique_intensity:
            channel_new[channel == i] = np.round(255 * (i / 255) ** inverse_cdf[i])

        new_channels.append(channel_new)
    # merging the three channels
    img_new = cv2.merge(new_channels)
    return img_new

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img


def display_side_by_side(original, processed):
    original = cv2.resize(original, (512, 512))
    processed = cv2.resize(processed, (512, 512))

    original = np.clip(original, 0, 255).astype(np.uint8)
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_rgb)
    plt.title("Processed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image = cv2.resize(image, (512, 512))
    processed_image, _ = remove_pip_black(image)
    image_with_agcwd = image_agcwd_color(processed_image)
    sharpened_image = sharpen_image(image_with_agcwd)
    display_side_by_side(image, sharpened_image)

if __name__ == "__main__":
    image_path = "img.jpg"
    main(image_path)