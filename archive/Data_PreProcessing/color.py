import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing  
from scipy.linalg import fractional_matrix_power

# Define the directory path
input_dirs = ['../Dataset_new/Dataset/training', '../Dataset_new/Dataset/validation']
output_dirs = ['../Dataset_try/training', '../Dataset_try/validation']


for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def image_agcwd(img, a=0.25, truncated_cdf=False):
    h, w = img.shape[:2]
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    intensity_max = unique_intensity.max()
    intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()

    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
    prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255) ** inverse_cdf[i])

    return img_new

def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed_img = 255 - agcwd
    return reversed_img

def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd

'''
def sepia(image):
   sepia_filter = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image
We use the above one to sepia our images
I do not think we need to sepia our images, but we can use the above template to provide a color filter over the images (we can use it to grayscale images)
'''

def sharpenn(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def adjust_brightness(image, brightness=30):
    bright_image = cv2.convertScaleAbs(image, beta=brightness)
    return bright_image

def adjust_contrast(image, contrast=1.5):
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return adjusted_image

'''
check sharpened image example here: '..\examples\sharpen.jpg' and decide whether to use
check brightness example here: '..\examples\brightness.jpg' and toggle around to decide on parameters
check contrast toggling here: ''..\examples\contrast.jpg'' and decide on parameters
check GBC toggling here: ''..\examples\gaussian_blur.jpg'' and decide on parameters
'''

def gaussian_blur_correction(image):
    kernel_size = (5, 5)  
    sigma = 10e-5
    kernel = cv2.getGaussianKernel(ksize=1, sigma=sigma)
    kernel = np.outer(kernel, kernel.T)
    corrected_image = cv2.filter2D(image, -1, kernel)
    return corrected_image

def apply_filter(image):
    YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]
    
    threshold = 0.3
    exp_in = 112  # Expected global average intensity
    M, N = image.shape[:2]
    mean_in = np.sum(Y / (M * N))
    t = (mean_in - exp_in) / exp_in

    if t < -threshold:  # Dimmed Image
        print("Dimmed Image Detected")
        Y = process_dimmed(Y)
    elif t > threshold:  # Bright Image
        print("Bright Image Detected")
        Y = process_bright(Y)

    YCrCb[:, :, 0] = Y
    image = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    #image = sepia(image)
    image = sharpenn(image)
    #image = adjust_brightness(image, brightness=50)    
    image = adjust_contrast(image, contrast=1.4)
    #image = gaussian_blur_correction(image)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image
'''
We are putting all our images through an IAGC layer and then sharpenning and increasing contrast
Then we put it through a Fast Non Local Means Denoising layer and then we return the image
'''

def process_image(filepath, output_directory):
    image = cv2.imread(filepath)
    print(filepath)
    if image is None:
        print(f"Error loading image {filepath}, skipping...")
        return
    
    filtered_image = apply_filter(image)
    
    relative_path = os.path.relpath(filepath, os.path.dirname(output_directory))
    output_path = os.path.join(output_directory, relative_path)
    output_subfolder = os.path.dirname(output_path)
    
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    success = cv2.imwrite(output_path, filtered_image)
    
    if success:
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")

def process_images_in_parallel(input_directory, output_directory):
    image_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))  
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        executor.map(process_image, image_files, [output_directory] * len(image_files))

if __name__ == "__main__":
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        print(input_dir, output_dir)
        process_images_in_parallel(input_dir, output_dir)
    print("Processing complete!")



'''
here we use Improved Adaptive Gamma Correction
Code courtesy https://github.com/leowang7/iagcwd/blob/master/IAGCWD.py 
Paper https://arxiv.org/abs/1709.04427

At the end of this, you can see that your new files would be saved in Dataset_new/Dataset/(training or validation)
If you can suggest a way out do let me know pls
'''