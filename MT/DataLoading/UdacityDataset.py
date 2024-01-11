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



class UdacityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, select_camera=None, slice_frames=None, select_ratio=1.0, select_range=None, optical_flow=True, seq_len=0, img_size=(224, 224)):
        
        assert select_ratio >= -1.0 and select_ratio <= 1.0 # positive: select to ratio from beginning, negative: select to ration counting from the end
        self.seq_len = seq_len
        camera_csv = pd.read_csv(csv_file)
        if select_camera:
            assert select_camera in ['left_camera', 'right_camera', 'center_camera'], "Invalid camera: {}".format(select_camera)
            camera_csv = camera_csv[camera_csv['frame_id']==select_camera]
        self.img_size = img_size
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0] # empty dataframe
            for start_idx in range(0, csv_len, slice_frames):
                if select_ratio > 0:
                    end_idx = int(start_idx + slice_frames * select_ratio)
                else:
                    start_idx, end_idx = int(start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames

                if end_idx > csv_len:
                    end_idx = csv_len
                if start_idx > csv_len:
                    start_idx = csv_len
                csv_selected = csv_selected.append(camera_csv[start_idx:end_idx])
            self.camera_csv = csv_selected
        elif select_range:
            csv_selected = camera_csv.iloc[select_range[0]: select_range[1]]
            self.camera_csv = csv_selected
        else:
            self.camera_csv = camera_csv
            
        self.root_dir = root_dir
        self.transform = transform
        self.optical_flow = optical_flow
        # Keep track of mean and cov value in each channel
        self.mean = {}
        self.std = {}
        for key in ['angle', 'torque', 'speed']:
            self.mean[key] = np.mean(camera_csv[key])
            self.std[key] = np.std(camera_csv[key])

    def __len__(self):
        return len(self.camera_csv)
    
    def read_data_single(self, idx):
        path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25,:,:]
        original_img = image.copy()

        angle = self.camera_csv['angle'].iloc[idx]
        angle_t = torch.tensor(angle)#.clamp(-1, 1)


        if self.transform:
            image_transformed = self.transform(cv2.resize(image, tuple(self.img_size)))

        if self.optical_flow:
            if idx != 0:
                path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx - 1])
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
            speed = self.camera_csv['speed'].iloc[idx]
            speed_t = torch.tensor(speed)
            return image_transformed, angle_t, optical_rgb, speed_t
        
        if self.transform:
            del image
            image = image_transformed
    
        return image, angle_t
    
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
            print('return read_data')
            data = None
            for i in idx:
                
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            if self.optical_flow:
                for stack_idx in [0, 1, 2, 3]: # we don't stack timestamp and frame_id since those are string data
                    #print('stack_idx', stack_idx, data[stack_idx]), data[3]=speed shape [5,]
                    data[stack_idx] = torch.stack(data[stack_idx])
            else:
                for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])

            return data
        
        else:
            print('return read_data single')
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.read_data(idx)

        sample = {'image': data[0],
                  'angle': data[1]}
        if self.optical_flow:
            sample['optical'] = data[2]
            sample['speed'] = data[3]
        
        del data
        
        return sample

