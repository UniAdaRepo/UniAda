#!/usr/bin/env python
# coding: utf-8


from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# defining customized Dataset class for Udacity
from .aug_utils import apply_augs
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random



class TestingDataset(Dataset):
    def __init__(self, name, transform=None, optical_flow=True, seq_len=0, img_size=(224, 224)):
        #assert select_ratio >= -1.0 and select_ratio <= 1.0 # positive: select to ratio from beginning, negative: select to ration counting from the end
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform
        self.optical_flow = optical_flow
        if name=='digital_Udacity_straight1':
            print(name)
            self.data_root = './digital_Udacity_straight1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name=='digital_Dave_curve1':
            print(name)
            self.data_root = './digital_Dave_curve1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name=='digital_Kitti_curve1':
            print(name)
            self.data_root = './digital_Kitti_curve1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.png')]
        elif name=='digital_Kitti_straight1':
            print(name)
            self.data_root = './digital_Kitti_straight1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.png')]
        elif name=='digital_Kitti_straight2':
            print(name)
            self.data_root = './digital_Kitti_straight2/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.png')]
        elif name=='digital_Dave_straight1':
            print(name)
            self.data_root = './digital_Dave_straight1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name=='udacity_curve1':
            print(name)
            self.data_root = './udacity_curve1/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name=='udacity_curve2':
            print(name)
            self.data_root = './udacity_curve2/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name == 'udacity_curve3':
            print(name)
            self.data_root = './udacity_curve3/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name == 'udacity_curve4':
            print(name)
            self.data_root = './udacity_curve4/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]
        elif name == 'udacity_straight4':
            print(name)
            self.data_root = './udacity_straight4/'
            for _, _, files in os.walk(self.data_root):
                print('read data')  # delete coordinates file
                files = files
            self.paths = [self.data_root + i for i in files if i.endswith('.jpg')]

        print('total imgs', len(self.paths))

    def __len__(self):
        return len(self.paths)
    def get_raw_imgs(self):
        raw_imgs = []
        for path in self.paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[65:-25, :, :]
            original_img = cv2.resize(image, tuple(self.img_size))
            raw_imgs.append(np.transpose(original_img,(2,0,1)))
        return np.array(raw_imgs)

    def read_data_single(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25,:,:] #?
        original_img = image.copy()

        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))

        if self.optical_flow:
            if idx != 0:
                path = self.paths[idx - 1]  #path for last timeframe
                prev = cv2.imread(path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25,:,:]
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            else:
                prev = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            cur = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            # Use Hue, Saturation, Value colour model
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            hsv = np.zeros(original_img.shape, dtype=np.uint8)
            hsv[..., 1] = 255
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            #optical_rgb, _ = apply_augs(optical_rgb, 0, augs, optical=True)
            optical_rgb = self.transform(cv2.resize(optical_rgb, tuple(self.img_size)))
            del original_img

            #print('image.max()', image_transformed.max(), 'min()', image_transformed.min())
            return image_transformed, optical_rgb
        
        if self.transform:
            del image
            image = image_transformed
    
        return image
    
    def read_data(self, idx):
        """
        Parameters
        ----------
        idx : list or int
            DESCRIPTION.
            in case of list:
                if len(idx) == batch_size -> do not choose augmentations since it will be applied to the whole batch
                if len(idx) == sequence_length -> apply augmentations
            in case of int:
                apply augmentations
        augs: a dict of augmentations
        Returns
        -------
        image(s), angle(s), (optical_flow: optional)
        """
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            if self.optical_flow:
                for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])
            else:
                for stack_idx in [0]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])

            return data
        
        else:
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.read_data(idx)

        sample = {'image': data[0]}
        if self.optical_flow:
            sample['optical'] = data[1]
        
        del data
        
        return sample

