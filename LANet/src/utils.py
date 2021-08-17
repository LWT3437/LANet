import cv2
import math
import numpy as np


def load_test_data(image_path, h=None, w=None):
    img = cv2.imread(image_path, -1).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if h != None and w != None:
        img = cv2.resize(img, (w, h))
    img = np.maximum(1.0, img)
    img = sRGB2linear(img)
    img = img * 2 - 1

    return img


def load_train_data(image_path, crop_scale=0.75, height=256, width=256):
    LDR = cv2.imread(image_path[0]).astype(np.float32)
    LDR = cv2.cvtColor(LDR, cv2.COLOR_BGR2RGB)
    HDR = cv2.imread(image_path[1], -1)
    HDR = cv2.cvtColor(HDR, cv2.COLOR_BGR2RGB)

    # crop the image
    h = LDR.shape[0]
    w = LDR.shape[1]
    if h / w < crop_scale:
        croped_w = int(h / crop_scale)
        start = int((w - croped_w) / 2)
        LDR = LDR[:, start:start+croped_w, :]
        HDR = HDR[:, start:start+croped_w, :]
    elif w / h < crop_scale:
        croped_h = int(w / crop_scale)
        start = int((h - croped_h) / 2)
        LDR = LDR[start:start+croped_h, :, :]
        HDR = HDR[start:start+croped_h, :, :]

    LDR = cv2.resize(LDR, (width, height))
    HDR = cv2.resize(HDR, (width, height))

    # pre-processing for LDR image
    LDR = np.maximum(1.0, LDR)
    LDR = sRGB2linear(LDR)
    LDR_I = np.mean(LDR, axis=2)
    mask = np.where(LDR_I > 0.83, 0, 1)  
    LDR = LDR * 2 - 1

    # pre-processing for HDR image
    HDR = np.maximum(HDR, 1e-8)
    HDR = np.minimum(HDR, 1e4) 
    HDR_I = np.mean(HDR, axis=2)
    mean_HDR = np.mean(mask * HDR_I)
    HDR = HDR * mean_LDR / mean_HDR
    HDR = np.log(HDR)

    train_pair = np.concatenate((LDR, HDR), axis=2)
    return train_pair


def save_images(images, size, image_path):
    if 'Msk_' in image_path:
        images = images * 255
    elif '_linear_' in image_path:
        images = (images + 1) / 2
    elif image_path.split('.')[-1] in ['jpg', 'png', 'bmp', 'tif']:
        images = (images + 1) / 2
        images = linear2sRGB(images)
    else:
        images = np.exp(images)

    cv2.imwrite(image_path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3), dtype=np.float32)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def rgb2hsi(RGB):
    R, G, B = np.split(RGB, 3, axis=2)

    I = np.mean(RGB, axis=2, keepdims=True)
    S = np.where(np.logical_and(R == G, G == B), 0, 1 -
                 np.minimum(R, np.minimum(G, B)) / I)
    v = np.where(np.logical_and(R == G, G == B), 0, ((R-G) + (R-B)) /
                 (2*np.sqrt(np.power(R-G, 2) + (R-B)*(G-B)) + 1e-8))
    v = np.minimum(v, 1)
    v = np.maximum(v, -1)
    v = np.arccos(v)
    H = np.where(B > G, 2*np.pi - v, v) / (2*np.pi)
    return np.concatenate([H, S, I], axis=2)


def hsi2rgb(HSI):
    H, S, I = np.split(HSI, 3, axis=2)
    H = H * 2*np.pi

    r = np.where(H < 2*np.pi/3, 1 + S*np.cos(H) / np.cos(np.pi/3 - H), 0)
    b = np.where(H < 2*np.pi/3, 1 - S, 0)
    g = np.where(H < 2*np.pi/3, 3 - r - b, 0)

    g = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3),
                 1 + S*np.cos(H-2*np.pi/3) / np.cos(np.pi - H), g)
    r = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), 1 - S, r)
    b = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), 3 - r - g, b)

    b = np.where(H >= 4*np.pi/3, 1 + S*np.cos(H-4*np.pi/3) /
                 np.cos(5*np.pi/3 - H), b)
    g = np.where(H >= 4*np.pi/3, 1 - S, g)
    r = np.where(H >= 4*np.pi/3, 3 - b - g, r)

    rgb = np.concatenate([r, g, b], axis=2)
    return I * rgb


def sRGB2linear(img):
    img = img / 255
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))


def linear2sRGB(img):
    img = np.where(img <= 0.0031308, img * 12.92,
                   np.power(img, 1/2.4) * 1.055 - 0.055)
    return img * 255
