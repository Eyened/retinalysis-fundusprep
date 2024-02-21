import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import sato


def rgb_to_luminance(image):
    return np.dot(image, [0.2126, 0.7152, 0.0722])


def convert_to_gray(self, image, method):
    if method == "luminance":
        return rgb_to_luminance(image)
    channel_index = {"red": 0, "green": 1, "blue": 2}[method]
    return image[:, :, channel_index]


def gamma_correction(image, gamma):
    return image**gamma


def to_uint8(image):
    return (255 * np.clip(image, 0, 1)).astype(np.uint8)


def vessel_enhance(image_gray, mask, sigmas=[9], gamma=0.8):
    filtered = sato(image_gray, sigmas=sigmas)
    filtered[~mask] = 0
    return gamma_correction(filtered, gamma)


def contrast_enhance(image, mask, sigma, contrast_factor=4):
    if len(image.shape) == 3:
        blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
    else:
        blurred = gaussian_filter(image, sigma=(sigma, sigma))
    ce = contrast_factor * (image - blurred) + 0.5
    ce = np.clip(ce, 0, 1)
    ce[~mask] = 0
    return ce


def CLAHE(image, cliplimit=2, tilesize=8):
    rgb = to_uint8(image)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image_clahe / 255


def combine_red_blue(img0, img1):
    if len(img0.shape) == 3:
        img0_gray = rgb_to_luminance(img0)
    else:
        img0_gray = img0

    if len(img1.shape) == 3:
        img1_gray = rgb_to_luminance(img1)
    else:
        img1_gray = img1

    img0 /= img0.max()
    img1 /= img1.max()

    h, w = img0_gray.shape
    combined_img = np.zeros((h, w, 3))
    combined_img[:, :, 0] = img1_gray
    combined_img[:, :, 1] = img0_gray
    combined_img[:, :, 2] = img0_gray

    return combined_img
