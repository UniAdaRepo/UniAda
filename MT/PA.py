
import sys
sys.path.append('./gradnorm/')
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DataLoading import TestingDataset as TD
from DataLoading import ConsecutiveBatchSampler as CB
from torchvision import transforms
from model.MotionTransformer import MotionTransformer
from easydict import EasyDict as edict
from tqdm import tqdm
import logging
import argparse


parser = argparse.ArgumentParser(description='forward hyperparameters')
parser.add_argument('--save_path', type=str, default='results/logger/Pert_Att/',
                        help='model savepath')
parser.add_argument('--direction', default="[-1,1]",type=str, help='testing task direction')
parser.add_argument('--beta', default=2.0, type=float, help='hyperparameter in loss function')
parser.add_argument('--eps', default=2.0, type=float, help='norm of perturbation: eps/255')
parser.add_argument('--title', default='digital_Udacity_straight1', type=str, help='title of the driving video')
args = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


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
raw_imgs = torch.tensor(test_set.get_raw_imgs()) #(22,3,224,224)
ori_imgs = [i['image'] for i in test_set] #transformed img
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


def denormalizes(im):
    for i in range(len(mean)):
        im.data[i] = im.data[i] * std[i] + mean[i]
    return im

def normalizes(im):
    for i in range(len(mean)):
        im.data[i] = (im.data[i] - mean[i]) / std[i]
    return im

def set_logger(log_path, name, run):
    logger = logging.getLogger(name+str(run))
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    return logger


def success_rate(mean_serror, mean_terror, logger):
    logger.info('--------------success rate in % on average----------------')
    mean_serror = mean_serror * 70
    mean_terror = mean_terror * 1.609344
    s_threshold = [3.5, 14, 21, 28]
    t_threshold = [4.6, 13.8, 23.0, 32.2]

    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} '.format(np.round(srate, 4), s))


    for t in t_threshold:
        trate = np.sum(mean_terror > t) / len(mean_terror) * 100
        logger.info('{0}% of frames have throttle error > {1} '.format(np.round(trate, 4), t))


'''
compute mean error and median error metrics
'''
iters = 4
alpha = 1/255
def compute_metric(serror_framelist, terror_framelist, logger):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(args.title, method, iters))
    ME_s = np.mean(serror_framelist)
    ME_t = np.mean(terror_framelist)
    logger.info('s={0}, t={1}'.format(ME_s * 70, ME_t * 1.609344))

    logger.info('-----median error {0} for {1}-----'.format(args.title, method))
    median_s = np.median(serror_framelist)
    median_t = np.median(terror_framelist)
    logger.info('s={0}, t={1}'.format(median_s * 70, median_t * 1.609344))


def pgd(logger, eps=2/255, iter_num=4, alpha=1/255, beta=2.0):
    d = eval(args.direction)
    s_adv, t_adv = [], []
    ori_imgs, ori_opticals = oris['ori_imgs'], oris['ori_opticals']
    logger.info('speed, steer')
    for i, (image, optical) in enumerate(zip(ori_imgs, ori_opticals)):
        img_ori = ori_imgs[i]
        ori_image_data = denormalizes(img_ori.clone().detach_().data)
        clean_image_data = denormalizes(img_ori.clone().detach_().data)
        image = Variable(img_ori, requires_grad=True)

        for iteration in range(iter_num):
            image.requires_grad = True
            angle_hat, speed_hat = network(image.repeat(5, 1, 1, 1).unsqueeze(0).cuda(), optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
            steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
            lossA, lossB = 1 / beta * torch.exp(steer_pred * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                speed_pred * (-1 / beta) * d[1])
            loss = 1/2 * lossA + 1/2 * lossB
            network.zero_grad()
            image.retain_grad()
            loss.backward(retain_graph=True)
            # denormalize image (->[0,1])
            image = denormalizes(image)
            #imgR = denormalize(imgR)

            # pgd attack
            adv_img_data = image.data - alpha * image.grad.sign()

            eta = torch.clamp(adv_img_data - clean_image_data, min=-eps, max=eps)
            #denormalize_ori_img (0,1)
            image = torch.clamp(ori_image_data + eta, min=0, max=1)
            image = normalizes(image).detach()
        s_adv.append(
            steer_pred.data.cpu().numpy())
        t_adv.append(speed_pred.data.cpu().numpy())
    return s_adv, t_adv
method = 'Perturb_att'
direction = eval(args.direction)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
for run in range(1):
    logger = set_logger(os.path.join(args.save_path, '{0}_{1}_{2}.log'.format(method, args.title, args.eps)), args.title, run)
    logger.info('method = {0}, direction = {1}, video = {2}, eps={3}'.format(method, args.direction, args.title, args.eps))
    s_adv, t_adv = pgd(logger=logger, eps=args.eps/255, iter_num=iters, alpha=alpha, beta=args.beta)
    serror = (np.array(s_adv)-ori_steer_pred) * direction[0]
    terror = (np.array(t_adv)-ori_speed_pred) * direction[1]
    logger.info('compute single run metric')
    compute_metric(serror, terror, logger)
    success_rate(serror, terror, logger)