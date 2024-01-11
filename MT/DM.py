import numpy as np
import os
import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoading import TestingDataset as TD
from DataLoading import ConsecutiveBatchSampler as CB
from torch import nn
from torchvision import transforms
from model.MotionTransformer import MotionTransformer
from easydict import EasyDict as edict
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description='forward hyperparameters')
parser.add_argument('--save_path', type=str, default='results/logger/DeepManeuver/',
                        help='model savepath')
parser.add_argument('--direction', default="right",type=str, help='direction')
parser.add_argument('--eps', default=2.0, type=float, help='norm of perturbation: eps/255')
parser.add_argument('--title', default='digital_Udacity_straight1', type=str, help='title of the driving video')
args = parser.parse_args()
title = args.title
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
noises = [15]
iter = 400

normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Normalize(*normalization)])

device = torch.device("cuda")
parameters = edict(
    batch_size = 1,
    seq_len = 5,
    num_workers = 4,
    model_name = 'MotionTransformer',
    normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    image_size=(224, 224),
    all_frames=True,
    optical_flow=True,
    checkpoint='./saved_models/transformer/opticaltransformer.tar'
)
model_object = MotionTransformer
network = model_object(parameters.seq_len)
network.load_state_dict(torch.load(parameters.checkpoint))
print('model loaded ... from', parameters.checkpoint)
network.to(device)
network.eval()
print('video name', args.title)
test_set = TD.TestingDataset(args.title, transform=transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(*parameters.normalization)
    ]), optical_flow=parameters.optical_flow, seq_len=parameters.seq_len,
                                 img_size=parameters.image_size)
raw_imgs = torch.tensor(test_set.get_raw_imgs())
ori_imgs = [i['image'] for i in test_set]
ori_opticals = [i['optical'] for i in test_set]
validation_cbs = CB.ConsecutiveBatchSampler(data_source=test_set, batch_size=parameters.batch_size,
                                            use_all_frames=True, shuffle=False, drop_last=False,
                                            seq_len=parameters.seq_len)
validation_loader = DataLoader(test_set, sampler=validation_cbs, num_workers=parameters.num_workers,
                               collate_fn=(lambda x: x[0]))

#get original predictions:
oris = {}
ori_steer_pred, ori_speed_pred = [], []
with torch.no_grad():
    for Validation_sample in tqdm(validation_loader):
        param_values = [v for v in Validation_sample.values()]
        image, optical = param_values
        optical = optical.to(device)
        image = image.to(device)
        angle_hat, speed_hat = network(image, optical)
        steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
        ori_steer_pred.append(steer_pred.data.cpu().numpy())
        ori_speed_pred.append(speed_pred.data.cpu().numpy())
oris['steer'], oris['speed'] = np.array(ori_steer_pred), np.array(ori_speed_pred)
oris['ori_imgs'], oris['ori_opticals'], oris['raw_imgs'] = ori_imgs, ori_opticals, raw_imgs

def set_logger(log_path, name, run):
    logger = logging.getLogger(name+str(run))
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger

class DeepManeuver():
    def __init__(self, model, seqpath, direction):
        self.model = model
        self.model = model
        self.seqpath = seqpath
        self.direction = direction
        self.sample_dir = os.getcwd() + "/sampledir"
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

    def perturb_images(self, dict, model: nn.Module, steering_vector: torch.Tensor,
                       bb_size=5, iterations=400, noise_level=25, device=torch.device("cuda"),
                       last_billboard=None, loss_fxn="MDirE", input_divers=False):
        img_arr, ori_opticals = dict['ori_imgs'], dict['ori_opticals']
        pert_shape = img_arr[0].shape

        model = model.to(device)
        steering_vector = steering_vector.to(device)

        perturbation = (torch.ones(1, *pert_shape)-0.5).float().to(device) #todo: change to 0.99999 rerun CILR&CILRS, 125 perturb too much!
        for i in range(iterations):
            perturbation = perturbation.detach()
            perturbation.requires_grad = True

            imgs = dict['raw_imgs'].float().to(device)
            imgs = imgs.to(device)
            perturbation_warp = torch.vstack([perturbation for _ in range(len(imgs))])
            imgs += perturbation_warp * 255

            imgs = torch.clamp(imgs/255 + torch.randn(*imgs.shape).to(device) / noise_level, 0, 1)
            imgs = transform(imgs)
            adv_steer = []
            loss, count = 0, 0
            for img, optical in zip(imgs, ori_opticals):
                angle_hat, _ = model(img.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                             optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
                steer_pred = angle_hat.flatten()[-1]
                loss += steer_pred - 1
                count += 1
                adv_steer.append(steer_pred)

            if self.direction == "left" and loss_fxn == "MDirE":
                tot_loss = -loss / count
            elif self.direction == "right" and loss_fxn == "MDirE":
                tot_loss = loss/count
            tot_loss.backward()
            # total perturb
            perturbation = torch.clamp(
                perturbation - torch.sign(perturbation.grad) / 1000, 0, args.eps/255
            )
            model.zero_grad()
        return torch.tensor(adv_steer)

def deepmaneuver(model, oris, direction, steering_vector, bb_size=5, iterations=400, noise_level=25,
                       dist_to_bb=None, last_billboard=None, input_divers=True, loss_fxn='inv23'):

    sdbb = DeepManeuver(model, None, direction)
    tensorized_steering_vector = torch.as_tensor(steering_vector, dtype=torch.float)
    adv_steer = sdbb.perturb_images(oris, model,
                            tensorized_steering_vector, bb_size=bb_size,
                            iterations=iterations, noise_level=noise_level,
                            last_billboard=last_billboard, loss_fxn=loss_fxn, input_divers=input_divers)
    return adv_steer

def success_rate(mean_serror, logger):
    logger.info('--------------success rate in % on average----------------')
    mean_serror = mean_serror * 70
    s_threshold = [3.5, 14, 21, 28]
    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} '.format(np.round(srate, 4), s))

def compute_metric(serror_framelist, logger):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(title, 'DM', iter))
    ME_s = np.mean(serror_framelist)
    logger.info('s={0}'.format(ME_s * 70))
    logger.info('-----median error {0} for {1}-----'.format(title, 'DM'))
    median_s = np.median(serror_framelist)
    logger.info('s={0}'.format(median_s * 70))

for noise in noises:
    direction = args.direction
    steering_vector2 = np.ones(len(ori_imgs))
    s_adv = deepmaneuver(network, oris, direction, copy.deepcopy(steering_vector2),
                                                       bb_size=5, iterations=iter, noise_level=noise,
                                                       dist_to_bb=1, input_divers=False, loss_fxn="MDirE")


    logger = set_logger(os.path.join(args.save_path, '{0}_{1}_{2}.log'.format('DeepManeuver', title, args.eps)), title, noise)
    logger.info('method = {0}, direction = {1}, video = {2}, eps={3}, noise={4}, iter={5}'.format('DeepManeuver', args.direction, title, args.eps, noise, iter))
    serror = (s_adv.numpy()-ori_steer_pred) * (-1)
    logger.info('compute single run metric')
    compute_metric(serror, logger)
    success_rate(serror, logger)