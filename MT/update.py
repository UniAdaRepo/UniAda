import os
import time
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.MotionTransformer import MotionTransformer
from DataLoading import TestingDataset as TD
from DataLoading import ConsecutiveBatchSampler as CB
import numpy as np
from FGSM import run_fgsm
from Universal import multi_gradnorm, multi
from DeepBillboard import deepbillboard
from metrics import success_rate, compute_metric
import logging
torch.cuda.set_device(0)
import argparse

parser = argparse.ArgumentParser(description='forward hyperparameters')
parser.add_argument('--save_path', type=str, default='results/logger/',
                        help='model savepath')
parser.add_argument('--eps', default=5.0, type=float, help='norm of perturbation: eps/255')
parser.add_argument('--scale', default=15.0, type=float, help='speed/scale')
parser.add_argument('--lr', default=2/255, type=float, help='learning rate for image gradients')
parser.add_argument('--bs', default=5, type=int, help='training batch size')
parser.add_argument('--lrgrad', default=0.4, type=float, help='learning rate for Dynamic weights learning')
parser.add_argument('--beta', default=2.0, type=float, help='hyperparameter in loss function')
parser.add_argument('--norm_threshold', default=0.3, type=float, help='threshold for image gradients norm')
parser.add_argument('--runs', default=1, type=int, help='number of times to train')
parser.add_argument('--iters', default=250, type=int, help='number of epochs in the training')
parser.add_argument('--alpha', default=10.0, type=float, help='hyperparmeter for Dynamic weights method')
parser.add_argument('--method', default='FGSM', type=str, help='method: "uni-const", "uni-dynamic", "FGSM", "deepbillboard"')
parser.add_argument('--title', default='digital_Udacity_straight1', type=str, help='title of the video data')
parser.add_argument('--times', default=1000, type=int, help='number of random weights in FGSM grid search')
parser.add_argument('--direction', default="[-1,1]",type=str, help='testing task direction')
parser.add_argument('--strategy', default='mean', type=str, help='strategy for combining pertubations in universal methods:"mean" or "max"')
#names = ['digital_Dave_curve1', 'digital_Dave_straight1', 'digital_Kitti_curve1', 'digital_Kitti_straight1',
#             'digital_Kitti_straight2', 'digital_Udacity_straight1']
args = parser.parse_args()
method = args.method
args.direction = eval(args.direction)
print('args.direction', args.direction)
print('method', method)

path = os.path.join(args.save_path,'{0} result/{1}'.format(method, args.title))
if not os.path.exists(path):
    os.makedirs(path)

def set_logger(log_path, name, run):
    logger = logging.getLogger(name+str(run))
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(logging.Formatter('%(message)s'))
    #logger.addHandler(stream_handler)

    return logger

'''Setup logger'''
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

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

def training_uni(hyper, model):
    print('our hyperparameter', hyper)
    title = hyper['title']
    method = hyper['method']
    runs = hyper['runs']
    d = hyper['direction']

    total_tadv, total_sadv = [], []
    for i in range(runs):
        logger = set_logger(
            os.path.join(args.save_path, '{0}_{1}_{2}_{3}.log'.format(method, args.weights, args.title, args.eps)), args.title, i)
        logger.info(
            'method = {0}, weights={1}, direction = {2}, video = {3}, eps={4}, lrgrad={5}, lr={6}, alpha={7}, beta={8}, iters={9}, scale={10}'.format(
                method, None,
                args.direction,
                args.title, args.eps, args.lrgrad, args.lr, args.alpha, args.beta, args.iters, args.scale))
        if method == 'uni-dynamic':
            perturb_updatelist, adv_vec = multi_gradnorm(model, oris, hyper, logger)
            np.save('./perturb/perturbation+{0}+eps+{1}+.npy'.format(args.title, args.eps), perturb_updatelist[-1].numpy())

        elif method == 'uni-const':
            perturb_updatelist, adv_vec = multi(model, oris, hyper, logger)

        elif method == 'deepbillboard':
            perturb_updatelist, adv_vec = deepbillboard(model, oris, hyper)
        else:
            raise Exception('Error: No method was given!')

        s_adv, t_adv = adv_vec[-1][0], adv_vec[-1][1]
        serror, terror = (s_adv-ori_steer_pred) * d[0], (t_adv-ori_speed_pred) * d[1]
        logger.info('compute single run metric')
        compute_metric(hyper, serror, terror, logger)
        success_rate(serror, terror, logger)
        total_sadv.append(s_adv)
        total_tadv.append(t_adv)
    mean_sadv = np.mean(total_sadv, axis=0) # mean over 5 runs
    mean_tadv = np.mean(total_tadv, axis=0)
    mean_serror, mean_terror = (mean_sadv - ori_steer_pred) * d[0], (mean_tadv - ori_speed_pred) * d[1]
    return mean_serror, mean_terror

start = time.time()
if method =='uni-dynamic' or method=='uni-const':
    hyper = {}
    hyper['method'] = method
    hyper['direction'] = args.direction
    hyper['strategy'] = args.strategy
    hyper['title'] = args.title
    hyper['runs'] = args.runs
    hyper['eps'] = args.eps
    hyper['bs'] = args.bs
    hyper['beta'] = args.beta
    hyper['iters'] = args.iters
    hyper['alpha'] = args.alpha
    hyper['norm_threshold'] = args.norm_threshold
    hyper['lr'] = args.lr
    hyper['lrgrad'] = args.lrgrad
    hyper['scale'] = args.scale
    if method == 'uni-const':
        hyper['weights'] = [1/2, 1/2]
    mean_serror, mean_terror = training_uni(hyper, network)

if method=='FGSM':
    hyper2 = {}
    hyper2['method'] = method
    hyper2['weight_strategy'] = 'equal'
    hyper2['eps'] = args.eps
    hyper2['lr'] = args.eps/255
    hyper2['direction'] = args.direction
    hyper2['beta'] = args.beta
    hyper2['iters'] = args.iters
    hyper2['title'] = args.title

    print('our hyperparameter', hyper2)
    for i in range(args.runs):
        print('run', i)
        logger = set_logger(
            os.path.join(args.save_path, '{0}_{1}_{2}_{3}.log'.format(method, args.weights, args.title, args.eps)), args.title, i)
        logger.info(
            'method = {0}, weights={1}, direction = {2}, video = {3}, eps={4}, lrgrad={5}, lr={6}, alpha={7}, beta={8} iterations={9}'.format(
                method, args.weights,
                args.direction,
                args.title, args.eps, args.lrgrad, args.lr, args.alpha, args.beta,
            args.iters))
        serror_framelist, terror_framelist = run_fgsm(hyper2, network, oris, logger)
        print('serror *70', np.mean(serror_framelist) * 70, 'terror * 1.609344', np.mean(terror_framelist) * 1.609344)
        logger.info('compute single run metric')
        compute_metric(hyper2, serror_framelist, terror_framelist, logger)
        success_rate(np.array(serror_framelist), np.array(terror_framelist), logger)



if method == 'deepbillboard':
    hyper3 = {}
    hyper3['iters'] = 250
    hyper3['jsma'] = True
    hyper3['strategy'] = args.strategy
    hyper3['direction'] = args.direction
    hyper3['bs'] = 5
    hyper3['simulated_annealing'] = True
    hyper3['eps'] = args.eps
    hyper3['sa_k'] = 30
    hyper3['sa_b'] = 0.96
    hyper3['runs'] = args.runs
    #hyper3['lr'] = args.lr
    hyper3['method'] = method

    hyper3['title'] = args.title
    hyper3['jsma_n'] = 3*200*200#42800
    hyper3['lr'] = args.lr
    hyper3['beta'] = args.beta
    mean_serror, mean_terror = training_uni(hyper3, network)



