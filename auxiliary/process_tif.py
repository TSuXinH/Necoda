import numpy as np
import cv2

CONVERT_UINT16_FLOAT64 = 65535.
SINGLE_TIFF_CONTENT_UPLIMIT_BYTE = 4e9


def tif2ndarray_min_max(tif_arr):
    """ Convert tif file to ndarray, and change the precision from uint16 to float32 for training. """
    tif_arr = tif_arr.astype(np.float32)
    tif_max = np.max(tif_arr)
    tif_min = np.min(tif_arr)
    np_arr = (tif_arr - tif_min) / (tif_max - tif_min)
    np_arr = np_arr.astype(np.float32)
    if tif_min < 0.:
        tif_max += -tif_min
        tif_min = 0.
    return np_arr, tif_max, tif_min


def tif2ndarray_mean(tif_arr):
    tif_arr = tif_arr.astype(np.float32)
    tif_mean = np.mean(tif_arr)
    np_arr = tif_arr - tif_mean
    return np_arr, tif_mean


def tif2ndarray_mean_std(tif_arr):
    tif_arr = tif_arr.astype(np.float32)
    tif_mean = np.mean(tif_arr)
    tif_std = np.std(tif_arr)
    np_arr = (tif_arr - tif_mean) / tif_std
    return np_arr, tif_mean, tif_std


def tif2ndarray_mean_max(tif_arr):
    tif_arr = tif_arr.astype(np.float32)
    mean = np.mean(tif_arr)
    max_minus_mean = np.max(tif_arr) - mean
    if max_minus_mean == 0:
        return np.zeros(tif_arr.shape)
    normalized_video = (tif_arr - mean) / max_minus_mean
    return normalized_video, mean, max_minus_mean


def ndarray2tif_min_max_clip(np_arr, tif_max=CONVERT_UINT16_FLOAT64, tif_min=0):
    " Convert ndarray to tif file, and change the precision from float32 to uint16, usually. "
    tif_arr = np_arr * (tif_max - tif_min) + tif_min
    # tif_arr -= np.min(tif_arr)
    tif_arr = tif_arr
    return tif_arr


def ndarray2tif_mean_clip(np_arr, tif_mean):
    tif_arr = np_arr + tif_mean
    tif_arr[tif_arr < 0] = 0
    tif_arr[tif_arr > CONVERT_UINT16_FLOAT64] = CONVERT_UINT16_FLOAT64
    return tif_arr


def ndarray2tif_mean_max_clip(np_arr, tif_mean, tif_max_minus_mean):
    tif_arr = np_arr * tif_max_minus_mean + tif_mean
    tif_arr = np.clip(tif_arr, 0, 65535)
    return tif_arr


def ndarray2tif_mean_std_clip(np_arr, tif_mean, tif_std):
    tif_arr = np_arr * tif_std + tif_mean
    tif_arr = np.clip(tif_arr, 0, 65535)
    return tif_arr


def center_crop_ndarray(np_arr, crop_size):
    h, w = np_arr.shape[-2], np_arr.shape[-1]
    h_crop, w_crop = crop_size
    if len(np_arr.shape) == 3:
        return np_arr[:, h // 2 - h_crop // 2: h // 2 + h_crop // 2, w // 2 - w_crop // 2: w // 2 + w_crop // 2]
    elif len(np_arr.shape) == 4:
        return np_arr[:, :, h // 2 - h_crop // 2: h // 2 + h_crop // 2, w // 2 - w_crop // 2: w // 2 + w_crop // 2]
    

def cal_tif_num(h, w, f, dtype='uint16'):
    if dtype == 'uint16':
        f_single = np.floor(SINGLE_TIFF_CONTENT_UPLIMIT_BYTE / h / w / 2)
    if dtype == 'single':
        f_single = np.floor(SINGLE_TIFF_CONTENT_UPLIMIT_BYTE / h / w / 4)
    tif_num = np.ceil(f / f_single)
    return tif_num, f_single


def video_resize(raw_video, new_x, new_y):
    t, _, _ = raw_video.shape
    raw_video = raw_video.astype(np.float64)
    new_video = np.zeros([t, new_y, new_x])
    for i in range(t):
        # print(raw_video[i].shape)
        new_video[i] = cv2.resize(raw_video[i], (new_x, new_y), interpolation=cv2.INTER_CUBIC)
    return new_video
