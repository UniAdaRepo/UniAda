import numpy as np
import os, random, copy
#import kornia
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

class DataSequence(data.Dataset):
    def __init__(self, img_array, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_arr = img_array
        self.transform = transform

    def __len__(self):
        return len(self.img_arr)

    def __getitem__(self, idx):
        sample = np.array(self.img_arr[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample

class DeepManeuver():

    def __init__(self, model, seqpath, direction):
        self.model = model
        self.model = model
        self.seqpath = seqpath
        self.direction = direction
        self.sample_dir = os.getcwd() + "/sampledir"
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

    def stripleftchars(self, s):
        for i in range(len(s)):
            if s[i].isnumeric():
                return s[i:]
        return -1

    def draw_arrows(self, img, angle1, angle2=None):
        import cv2

        img = (img.transpose((1, 2, 0)) * 255).round().astype(np.uint8).copy()

        pt1 = (int(img.shape[1] / 2), img.shape[0])
        pt2_angle1 = (
            int(img.shape[1] / 2 - img.shape[0] / 3 * np.sin(angle1)),
            int(img.shape[0] - img.shape[0] / 3 * np.cos(angle1)),
        )
        img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 3)
        if angle2 is not None:
            angle2 = -angle2
            pt2_angle2 = (
                int(img.shape[1] / 2 - img.shape[0] / 3 * np.sin(angle2)),
                int(img.shape[0] - img.shape[0] / 3 * np.cos(angle2)),
            )
            img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 3)
        return img.astype(np.float32).transpose((2, 0, 1)) / 255

    def input_diversification(self, imgs, y_orig, device):
        imgs_rs = imgs
        y_orig_exp = copy.deepcopy(y_orig)
        for i in range(int(imgs.shape[0] / 2)):
            r1 = random.randint(0, imgs.shape[0]-1)
            r2 = random.uniform(0.5, 1.5)
            scale_factor = torch.tensor([[r2]]).float().to(device)
            temp = kornia.geometry.transform.scale(imgs[r1][None], scale_factor)
            imgs_rs = torch.cat((imgs_rs,temp))
            y_orig_exp = torch.cat((y_orig_exp, y_orig[r1][None]),dim=0)
        return imgs_rs, y_orig_exp


    def perturb_images(self, dict, model: nn.Module, steering_vector: torch.Tensor,
                       bb_size=5, iterations=400, noise_level=25, device=torch.device("cuda"),
                       last_billboard=None, loss_fxn="MDirE", input_divers=False):
        img_arr, ori_speed, ori_controls = dict['imgs'], dict['speed'], dict['controls']
        pert_shape = c,h,w = img_arr.shape[1:]

        model = model.to(device)
        steering_vector = steering_vector.to(device)
        dataset = DataSequence(img_arr)
        data_loader = data.DataLoader(dataset, batch_size=len(dataset))
        shape = next(iter(data_loader)).shape[2:]
        orig_shape = shape


        perturbation = (torch.ones(1, *pert_shape)-0.5).float().to(device)
        for i in range(iterations):
            perturbation = perturbation.detach()
            perturbation.requires_grad = True
            imgs = next(iter(data_loader)).to(device)
            perturbation_warp = torch.vstack([perturbation for _ in range(len(imgs))])


            imgs += perturbation_warp

            imgs = torch.clamp(imgs + torch.randn(*imgs.shape).to(device) / noise_level, 0, 1)
            ori_output, _ = model.forward_branch(imgs, ori_speed.cuda(), ori_controls)
            y = ori_output[:,0]

            if self.direction == "left" and loss_fxn == "MDirE":
                loss = (y.flatten() - steering_vector).mean()
            elif self.direction == "right" and loss_fxn == "MDirE":
                loss = -(y.flatten() - steering_vector).mean()
            elif self.direction == "straight" and loss_fxn == "MDirE":
                loss = (y.flatten() - steering_vector).mean()
            elif loss_fxn == "MSE":
                loss = F.mse_loss(y.flatten(), steering_vector)
            elif loss_fxn == "MAbsE":
                loss = abs(y.flatten() - steering_vector).mean()
            elif loss_fxn == "inv23" and self.direction == "left":
                decay_factors = 1 / torch.pow(torch.Tensor([i for i in range(1, y.flatten().shape[0] + 1)]), 1/100).to(device)
                loss = (decay_factors * (y.flatten() - steering_vector)).mean()
            elif loss_fxn == "inv23" and self.direction == "right":
                decay_factors = 1 / torch.pow(torch.Tensor([i for i in range(1, y.flatten().shape[0] + 1)]), 1/100).to(device)
                loss = -(decay_factors * (y.flatten() - steering_vector)).mean()
            elif loss_fxn == "inv23" and self.direction == "straight":
                decay_factors = 1 / torch.pow(torch.Tensor([i for i in range(1, y.flatten().shape[0] + 1)]), 1/10).to(device)
                loss = (decay_factors * (y.flatten() - steering_vector)).mean()

            print(
                f"[iteration {i:5d}/{iterations}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
            )
            loss.backward()

            perturbation = torch.clamp(
                perturbation - torch.sign(perturbation.grad) / 1000, 0, 2/255
            )
            model.zero_grad()
        print(
            f"[iteration {i:5d}/{iterations}] loss={loss.item():2.5f} max(angle)={y.max().item():2.5f} min(angle)={y.min().item():2.5f} mean(angle)={y.mean().item():2.5f} median(angle)={torch.median(y).item():2.5f}"
        )
        return y, perturbation
