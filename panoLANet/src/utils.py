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
    if height == width:
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
    mean_LDR = np.mean(mask * LDR_I)
    LDR = LDR * 2 - 1

    # pre-processing for HDR image
    HDR_I = np.mean(HDR, axis=2)
    mean_HDR = np.mean(mask * HDR_I)
    HDR = HDR * mean_LDR / mean_HDR
    HDR = np.maximum(HDR, 1e-8)
    HDR = np.minimum(HDR, 1e4) 
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


def sRGB2linear(img):
    img = img / 255
    return np.where(img <= 0.04045, img / 12.92, np.power((img+0.055) / 1.055, 2.4))


def linear2sRGB(img):
    img = np.where(img <= 0.0031308, img * 12.92,
                   np.power(img, 1/2.4) * 1.055 - 0.055)
    return img * 255


def p2c(h, fov=2*np.pi/3):

    grid = np.arange(1/h - 1, 1 + 1/h, 2/h)[::-1]
    yy, xx = np.meshgrid(grid, grid)

    d = np.tan(fov/2)   # the inverse of distance between camera focus and original point
    angle = np.arctan(np.sqrt(xx*xx+yy*yy) * d)     # angle between z axis and pixel's ray
    theta = angle + np.arcsin(np.sin(angle) / d / np.sqrt(2))  # the sum of the internal angles of the triangle is 180 degrees
    phi = np.where(xx > 0, np.mod(np.arctan(yy/xx), 2*np.pi), np.mod(np.arctan(yy/xx) + np.pi, 2*np.pi))

    row = theta*h / np.pi - 0.5
    row_t = row.astype(int)
    row_b = np.where(row <= h-1, np.ceil(row).astype(int), h-1)
    row_args = np.where(row >= 0, row - row_t, row_t - row)
    col = phi*h / np.pi - 0.5
    col_l = np.where(col >= 0, col.astype(int), 2*h - 1)
    col_r = np.mod(np.ceil(col).astype(int), 2*h)
    col_args = np.mod(col - col_l, 2*h)
    
    return [np.stack([row_t, col_l], axis=2).tolist(), np.stack([row_t, col_r], axis=2).tolist(), 
            np.stack([row_b, col_l], axis=2).tolist(), np.stack([row_b, col_r], axis=2).tolist(), 
            np.expand_dims(row_args, -1), np.expand_dims(col_args, -1)]
    

def c2p(h, fov=2*np.pi/3):
    w = 2 * h

    grid_row = np.arange(np.pi/(2*h), np.pi*(1+1/(2*h)), np.pi/h)
    grid_col = np.arange(np.pi/w, 2*np.pi*(1+1/(2*w)), np.pi/h)
    pp, tt = np.meshgrid(grid_col, grid_row)
    
    r = np.where(tt < np.pi/2, np.sqrt(2) * np.sin(tt) / (1 + np.sqrt(2)*np.cos(tt)*np.tan(fov/2)), 0)  # radius of intersection
    x = r * np.cos(pp)
    y = r * np.sin(pp)

    row = np.where(np.abs(x) < 1, np.where(np.abs(y) < 1, (1-x)*h/2 - 0.5, 0), 0)
    row_t = row.astype(int)
    row_b = np.where(row > h-1, h-1, np.ceil(row).astype(int))
    row_args = np.where(row >= 0, row - row_t, row_t - row)
    col = np.where(np.abs(x) < 1, np.where(np.abs(y) < 1, (1-y)*h/2 - 0.5, 0), 0)
    col_l = col.astype(int)
    col_r = np.where(col > h-1, h-1, np.ceil(col).astype(int))
    col_args = np.where(col >= 0, col - col_l, col_l - col)

    return [np.stack([row_t, col_l], axis=2).tolist(), np.stack([row_t, col_r], axis=2).tolist(), 
            np.stack([row_b, col_l], axis=2).tolist(), np.stack([row_b, col_r], axis=2).tolist(), 
            np.expand_dims(row_args, -1), np.expand_dims(col_args, -1)]


def c2p_mask(h, fov=2*np.pi/3):
    w = 2 * h

    grid_row = np.arange(np.pi/(2*h), np.pi*(1+1/(2*h)), np.pi/h)
    grid_col = np.arange(np.pi/w, 2*np.pi*(1+1/(2*w)), np.pi/h)
    pp, tt = np.meshgrid(grid_col, grid_row)
    
    r = np.where(tt < np.pi/2, np.sqrt(2) * np.sin(tt) / (1 + np.sqrt(2)*np.cos(tt)*np.tan(fov/2)), 0)  # radius of intersection
    x = r * np.cos(pp)
    y = r * np.sin(pp)

    return np.expand_dims(np.logical_and(tt < np.pi/2, np.logical_and(np.abs(x) < 1, np.abs(y) < 1)), -1)
