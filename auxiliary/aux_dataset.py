import os
import torch
import random
import math
import numpy as np
from torch import nn
import tifffile as tif
from torch.utils.data import Dataset

from .process_tif import tif2ndarray_min_max, tif2ndarray_mean, tif2ndarray_mean_std, tif2ndarray_mean_max, video_resize


def random_transform(input, p_trans):
    if p_trans == 0:  # no transformation
        input = input
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
    return input


class DatasetTifPatch(Dataset):
    def __init__(
            self,
            data_path,
            patch_x=128,
            patch_t=128,
    ):
        super().__init__()
        tif_arr = tif.imread(data_path)
        self.neural_data, self.tif_max, self.tif_min = tif2ndarray_min_max(tif_arr)
        self.x = self.neural_data.shape[1]
        self.y = self.neural_data.shape[2]
        self.t = self.neural_data.shape[0]
        self.patch_x = patch_x
        self.patch_y = patch_x
        self.patch_t = patch_t
        self.final_size = self.patch_x * self.patch_y * self.patch_t
        self.patch_num = self.num_x = self.num_y = self.num_t = 0
        self.create_patch_info()

        print('Finish creating dataset.')
        print('Shape: ', self.neural_data.shape)
        print('Tif min: ', self.tif_min)
        print('Tif max: ', self.tif_max)

    def __len__(self):
        return self.patch_num

    def __getitem__(self, item):
        patch_coordinate_x, patch_coordinate_y, patch_coordinate_t = get_patch_position(item, self.num_x)
        return {
            'patch': torch.FloatTensor(
                np.expand_dims(
                    self.neural_data[
                        patch_coordinate_t * self.patch_t: (patch_coordinate_t + 1) * self.patch_t,
                        patch_coordinate_x * self.patch_x: (patch_coordinate_x + 1) * self.patch_x,
                        patch_coordinate_y * self.patch_y: (patch_coordinate_y + 1) * self.patch_y
                    ], axis=0)
            ),
            'patch_id': item
        } if (patch_coordinate_t + 1) * self.patch_t < self.t else {
            'patch': torch.FloatTensor(
                np.expand_dims(
                    self.neural_data[
                        self.t - self.patch_t: self.t,
                        patch_coordinate_x * self.patch_x: (patch_coordinate_x + 1) * self.patch_x,
                        patch_coordinate_y * self.patch_y: (patch_coordinate_y + 1) * self.patch_y
                    ], axis=0)
            ),
            'patch_id': item
        }

    def create_patch_info(self):
        self.num_x = self.num_y = int(self.x // self.patch_x)
        self.num_t = int(self.t // self.patch_t)
        self.patch_num = self.num_x * self.num_y * self.num_t if self.t % self.patch_t == 0 else self.num_x * self.num_y * (
                self.num_t + 1)


def get_patch_position(idx, num_x):
    num_y = num_x
    patch_coordinate_t = idx // (num_x * num_y)
    coordinate_t_remainder = idx % (num_x * num_y)
    patch_coordinate_x = coordinate_t_remainder // num_y
    patch_coordinate_y = coordinate_t_remainder % num_y
    return patch_coordinate_x, patch_coordinate_y, patch_coordinate_t


class DatasetTifPatchWithOverlap(Dataset):
    def __init__(
            self,
            data_path,
            patch_x=128,
            patch_t=128,
            gap_x=64,
            gap_t=64,
            apply_aug=False,
            apply_sampling=False
    ):
        super().__init__()
        tif_arr = tif.imread(data_path)
        self.neural_data, self.tif_max, self.tif_min = tif2ndarray_min_max(tif_arr)
        self.x = self.neural_data.shape[-1]
        self.y = self.neural_data.shape[-2]
        self.t = self.neural_data.shape[0]
        self.patch_x = patch_x
        self.patch_y = patch_x
        self.patch_t = patch_t
        self.apply_aug = apply_aug
        self.apply_sampling = apply_sampling
        self.coordinate_list = create_overlap_patch_info_train(patch_x, gap_x, self.x, patch_t, gap_t, self.t)

        print('Finish creating dataset.')
        print('Shape: ', self.neural_data.shape)
        print('Tif min: ', self.tif_min)
        print('Tif max: ', self.tif_max)

    def __len__(self):
        return len(self.coordinate_list)

    def __getitem__(self, item):
        current_coordinate_list = self.coordinate_list[item]
        # print('current coordinate_list: ', current_coordinate_list)
        stack = self.neural_data[
                current_coordinate_list[2]: current_coordinate_list[2] + self.patch_t,
                current_coordinate_list[0]: current_coordinate_list[0] + self.patch_x,
                current_coordinate_list[1]: current_coordinate_list[1] + self.patch_y,
                ]
        (raw, target) = hierarchical_data_generation(stack) if self.apply_sampling else (stack.copy(), stack.copy())
        if self.apply_aug:
            p_trans = random.randrange(8)
            raw = random_transform(raw, p_trans)
            target = random_transform(target, p_trans)
        raw = np.expand_dims(raw, axis=0)
        target = np.expand_dims(target, axis=0)
        # print('raw.shape: ', raw.shape)
        # print('target.shape: ', target.shape)
        return {
            'patch': torch.FloatTensor(raw.copy()),
            'target': torch.FloatTensor(target.copy()),
            'patch_id': item
        }


class DatasetTifPatchTrainWithPadding(Dataset):
    def __init__(
            self,
            data_path,
            patch_x=128,
            patch_t=128,
            gap_x=64,
            gap_t=64,
            interp_x=4,
            interp_t=4,
            apply_aug=False,
            apply_sampling=False,
            pre_norm='min_max',
            do_resize=False,
            x_resize=0,
            y_resize=0,
    ):
        super().__init__()
        tif_arr = tif.imread(data_path)
        if do_resize:
            tif_arr = video_resize(tif_arr, x_resize, y_resize)
        if pre_norm == 'min_max':
            self.neural_data, self.tif_max, self.tif_min = tif2ndarray_min_max(tif_arr)
            print('Tif min: ', self.tif_min)
            print('Tif max: ', self.tif_max)
        elif pre_norm == 'mean_max':
            self.neural_data, self.tif_mean, self.tif_max_minus_mean = tif2ndarray_mean_max(tif_arr)
            print('Tif mean: ', self.tif_mean)
            print('Tif max minus mean: ', self.tif_max_minus_mean)
        elif pre_norm == 'mean_std':
            self.neural_data, self.tif_mean, self.tif_std = tif2ndarray_mean_std(tif_arr)
            print('Tif mean: ', self.tif_mean)
            print('Tif std: ', self.tif_std)
        else:
            raise NotImplementedError('Wrong pre-norm')
        self.t = self.neural_data.shape[0]
        self.x = self.neural_data.shape[1]
        self.y = self.neural_data.shape[2]
        self.patch_x = patch_x
        self.patch_y = patch_x
        self.patch_t = patch_t
        self.interp_x = self.interp_y = interp_x
        self.interp_t = interp_t
        self.apply_aug = apply_aug
        self.apply_sampling = apply_sampling
        self.coordinate_list = create_overlap_patch_info_train(patch_x, gap_x, self.x, patch_t, gap_t, self.t, self.y)

        print('Finish creating dataset.')
        print('Shape: ', self.neural_data.shape)

    def __len__(self):
        return len(self.coordinate_list)

    def __getitem__(self, item):
        current_coordinate_list = self.coordinate_list[item]
        start_t, end_t = get_interp_coord(current_coordinate_list[0], self.t, self.patch_t, self.interp_t)
        start_x, end_x = get_interp_coord(current_coordinate_list[1], self.x, self.patch_x, self.interp_x)
        start_y, end_y = get_interp_coord(current_coordinate_list[2], self.y, self.patch_y, self.interp_y)
        stack = self.neural_data[start_t: end_t, start_x: end_x, start_y: end_y]
        (raw, target) = hierarchical_data_generation(stack) if self.apply_sampling else (stack.copy(), stack.copy())
        if self.apply_aug:
            p_trans = random.randrange(8)
            raw = random_transform(raw, p_trans)
            target = random_transform(target, p_trans)
        raw = np.expand_dims(raw, axis=0)
        target = np.expand_dims(target, axis=0)
        return {
            'patch': torch.FloatTensor(raw.copy()),
            'target': torch.FloatTensor(target.copy()),
            'patch_id': item
        }


class DatasetTifPatchTest(Dataset):
    def __init__(
            self,
            data_path,
            patch_x=128,
            patch_t=128,
            interp_x=4,
            interp_t=4,
            pre_norm='min_max',
            do_resize=False,
            x_resize=0,
            y_resize=0,
    ):
        super().__init__()
        tif_arr = tif.imread(data_path)
        if do_resize:
            tif_arr = video_resize(tif_arr, x_resize, y_resize)
        if pre_norm == 'min_max':
            self.neural_data, self.tif_max, self.tif_min = tif2ndarray_min_max(tif_arr)
            print('Tif min: ', self.tif_min)
            print('Tif max: ', self.tif_max)
        elif pre_norm == 'mean_max':
            self.neural_data, self.tif_mean, self.tif_max_minus_mean = tif2ndarray_mean_max(tif_arr)
            print('Tif mean: ', self.tif_mean)
            print('Tif max minus mean: ', self.tif_max_minus_mean)
        elif pre_norm == 'mean_std':
            self.neural_data, self.tif_mean, self.tif_std = tif2ndarray_mean_std(tif_arr)
            print('Tif mean: ', self.tif_mean)
            print('Tif std: ', self.tif_std)
        else:
            raise NotImplementedError('Wrong pre-norm')
        self.t = self.neural_data.shape[0]
        self.x = self.neural_data.shape[1]
        self.y = self.neural_data.shape[2]
        self.patch_x = self.patch_y = patch_x
        self.patch_t = patch_t
        self.interp_x = self.interp_y = interp_x
        self.interp_t = interp_t
        self.final_size = self.patch_x * self.patch_y * self.patch_t
        self.patch_num = self.num_x = self.num_y = self.num_t = 0
        self.coordinate_list = create_overlap_patch_info_test(patch_x, self.x, patch_t, self.t, self.y)

        print('Finish creating dataset.')
        print('Shape: ', self.neural_data.shape)

    def __len__(self):
        return len(self.coordinate_list)

    def __getitem__(self, item):
        current_coordinate_list = self.coordinate_list[item]
        start_t, end_t = get_interp_coord(current_coordinate_list[0], self.t, self.patch_t, self.interp_t)
        start_x, end_x = get_interp_coord(current_coordinate_list[1], self.x, self.patch_x, self.interp_x)
        start_y, end_y = get_interp_coord(current_coordinate_list[2], self.y, self.patch_y, self.interp_y)
        stack = np.expand_dims(self.neural_data[start_t: end_t, start_x: end_x, start_y: end_y], axis=0)
        return {'patch': torch.FloatTensor(stack), 'patch_id': item}

    def create_patch_info(self):
        self.num_x = int(self.x // self.patch_x)
        self.num_y = int(self.y // self.patch_y)
        self.num_t = int(self.t // self.patch_t)
        self.patch_num = self.num_x * self.num_y * self.num_t if self.t % self.patch_t == 0 else self.num_x * self.num_y * (self.num_t + 1)


def create_overlap_patch_info_train(patch_x, gap_x, whole_x, patch_t, gap_t, whole_t, whole_y=0):
    patch_y = patch_x
    gap_y = gap_x
    whole_y = whole_x if whole_y == 0 else whole_y
    coordinate_list = []

    assert gap_y >= 0 and gap_x >= 0 and gap_t >= 0, "train gap size is negative!"
    print('gap_t -----> ', gap_t)
    print('gap_x -----> ', gap_x)
    print('gap_y -----> ', gap_y)

    for x in range(int((whole_x - patch_x + gap_x) / gap_x)):
        for y in range(int((whole_y - patch_y + gap_y) / gap_y)):
            for t in range(int((whole_t - patch_t + gap_t) / gap_t)):
                init_t = gap_t * t
                init_h = gap_x * x
                init_w = gap_y * y
                single_coordinate = [init_t, init_h, init_w]
                coordinate_list.append(single_coordinate)

    return coordinate_list


def create_overlap_patch_info_test(patch_x, whole_x, patch_t, whole_t, whole_y=0):
    patch_y = patch_x
    whole_y = whole_x if whole_y == 0 else whole_y
    coordinate_list = []

    for x in range(int(whole_x / patch_x)):
        for y in range(int(whole_y / patch_y)):
            for t in range(int(whole_t / patch_t)):
                init_t = patch_t * t
                init_x = patch_x * x
                init_y = patch_y * y
                single_coordinate = [init_t, init_x, init_y]
                coordinate_list.append(single_coordinate)
    if whole_t % patch_t != 0:
        for x in range(int(whole_x / patch_x)):
            for y in range(int(whole_y / patch_y)):
                init_x = patch_x * x
                init_y = patch_y * y
                single_coordinate = [whole_t - patch_t, init_x, init_y]
                coordinate_list.append(single_coordinate)
    return coordinate_list


def get_interp_coord(cur_start_coord, total_size, patch_size, interp_length):
    if cur_start_coord < interp_length:
        start_p = cur_start_coord
        end_p = cur_start_coord + patch_size + 2 * interp_length
    elif cur_start_coord + patch_size + interp_length > total_size:
        end_p = cur_start_coord + patch_size
        start_p = cur_start_coord - 2 * interp_length
    else:
        start_p = cur_start_coord - interp_length
        end_p = cur_start_coord + patch_size + interp_length
    return start_p, end_p


def random_pick(dim):
    rand_list = []
    for _ in range(dim):
        rand_list.append(np.random.randint(0, 2))
    return rand_list


def temporal_sampling(stack):
    rand_list = random_pick(1)
    rand0 = rand_list[0]
    rand1 = 1 - rand0
    raw = stack[rand0:: 2]
    target = stack[rand1:: 2]
    return raw, target


def single_dim_sampling(stack, dim=0):
    rand_list = random_pick(1)
    rand0 = rand_list[0]
    rand1 = 1 - rand0
    if dim == 0:
        raw = stack[rand0:: 2]
        target = stack[rand1:: 2]
    elif dim == 1:
        raw = stack[:, rand0:: 2]
        target = stack[:, rand1:: 2]
    elif dim == 2:
        raw = stack[:, :, rand0:: 2]
        target = stack[:, :, rand1:: 2]
    else:
        raise NotImplementedError
    return raw, target


def spatial_sampling(stack):
    raw_rand_list = random_pick(2)
    target_rand_list = random_pick(2)
    while raw_rand_list == target_rand_list:
        target_rand_list = random_pick(2)
    raw = stack[:, raw_rand_list[0]:: 2, raw_rand_list[1]:: 2]
    target = stack[:, target_rand_list[0]:: 2, target_rand_list[1]:: 2]
    return raw, target


def spatiotemporal_sampling(stack):
    raw_rand_list = random_pick(3)
    target_rand_list = random_pick(3)
    while raw_rand_list == target_rand_list:
        target_rand_list = random_pick(3)
    raw = stack[raw_rand_list[0]:: 2, raw_rand_list[1]:: 2, raw_rand_list[2]:: 2]
    target = stack[target_rand_list[0]:: 2, target_rand_list[1]:: 2, target_rand_list[2]:: 2]
    return raw, target


def hierarchical_data_generation(stack):
    # order = np.random.randint(0, 4)
    # if order == 0:
    #     raw = stack
    #     target = stack
    # elif order == 1:
    #     raw, target = temporal_sampling(stack)
    # elif order == 2:
    #     raw, target = spatial_sampling(stack)
    # else:
    #     raw, target = spatiotemporal_sampling(stack)
    # return (raw, target)
    raw, target = spatiotemporal_sampling(stack)
    return (raw, target)


def post_one_dim_split(patch_input):
    randomness = np.random.randint(0, 3)
    rand_start = np.random.randint(0, 2)
    if randomness == 0:  # sacrifice the t axis
        patch_data = patch_input[:, :, rand_start::2]
        patch_gt = patch_input[:, :, 1 - rand_start::2]
    elif randomness == 1:
        patch_data = patch_input[:, :, :, rand_start::2]
        patch_gt = patch_input[:, :, :, 1 - rand_start::2]
    elif randomness == 2:
        patch_data = patch_input[:, :, :, :, rand_start::2]
        patch_gt = patch_input[:, :, :, :, 1 - rand_start::2]
    else:
        raise ValueError('Randomness for the choice of axis to be sacrificed is not correct.')

    return patch_data, patch_gt, randomness
