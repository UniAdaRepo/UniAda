import os
import time
import torch
import numpy as np
from create_log_folder import create_folder
from FGSM import run_fgsm
from Universal import multi_gradnorm, multi
from upload.DeepBillboard import deepbillboard
from configs import g_conf, merge_with_yaml
from network import CoILModel
from input import Augmenter, coil_valid_dataset
from metrics import success_rate, compute_metric

torch.cuda.set_device(0)
import argparse

parser = argparse.ArgumentParser(description='forward hyperparameters')
parser.add_argument('--eps', default=2.0, type=float, help='norm of perturbation: eps/255')
parser.add_argument('--lr', default=2/255, type=float, help='learning rate for image gradients')
parser.add_argument('--bs', default=5, type=int, help='training batch size')
parser.add_argument('--lrgrad', default=0.4, type=float, help='learning rate for Dynamic weights learning')
parser.add_argument('--beta', default=2.0, type=float, help='hyperparameter in loss function')
parser.add_argument('--norm_threshold', default=0.3, type=float, help='threshold for image gradients norm')
parser.add_argument('--runs', default=1, type=int, help='number of times to train')
parser.add_argument('--iters', default=250, type=int, help='number of epochs in the training')
parser.add_argument('--alpha', default=10.0, type=float, help='hyperparmeter for Dynamic weights method')
parser.add_argument('--method', default='FGSM', type=str, help='method: "uni-const", "uni-dynamic", "FGSM", "deepbillboard"')
parser.add_argument('--title', default='zip3,00715,white_car,raining', type=str, help='title of the video data')
parser.add_argument('--weights', default='equal', type=str, help='grid search best, equal, steer, throttle, brake, must be string')
parser.add_argument('--times', default=1000, type=int, help='number of random weights in FGSM grid search')
parser.add_argument('--direction', default="[1,1,-1]",type=str, help='testing task direction')
parser.add_argument('--strategy', default='mean', type=str, help='strategy for combining pertubations in universal methods:"mean" or "max"')
args = parser.parse_args()
method = args.method
args.direction = eval(args.direction)
create_folder(method, args.title)

full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
augmenter = Augmenter(None)
merge_with_yaml(os.path.join('configs/nocrash/resnet34imnet10S1.yaml'))
model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
checkpoint = torch.load(os.path.join('./_logs/nocrash/resnet34imnet10S1/checkpoints/660000.pth'))
#load CILR model
#merge_with_yaml(os.path.join('./gradnorm/configs/nocrash/resnet34imnet10-nospeed.yaml'))
#model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
#checkpoint = torch.load(os.path.join('./gradnorm/_logs/nocrash/resnet34imnet10-nospeed/checkpoints/760000.pth'))
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()  # this is for turn off dropout

def training_uni(hyper, model, full_dataset, augmenter):
    title = hyper['title']
    method = hyper['method']
    runs = hyper['runs']
    d = hyper['direction']
    dataset_central = coil_valid_dataset.CoILDataset_central_valid(full_dataset, title, transform=augmenter,
                                                                   preload_name=None)
    full_loader = torch.utils.data.DataLoader(dataset_central, batch_size=len(dataset_central), shuffle=False)
    oris = {}
    with torch.no_grad():
        for full_data in full_loader:
            ori_speed = dataset_central.extract_inputs(full_data)
            ori_controls = full_data['directions']
            ori_output, _ = model.forward_branch(full_data['rgb'].cuda(), ori_speed.cuda(), ori_controls)
            oris['speed'], oris['controls'], oris['output'] = ori_speed, ori_controls, ori_output
            torch.cuda.empty_cache()
    ori_throttle_pred, ori_steer_pred = ori_output[:, 1].data.cpu().numpy(), ori_output[:,0].data.cpu().numpy()
    total_tadv, total_sadv, total_badv = [], [], []
    for i in range(runs):
        print('run',i)
        if method == 'uni-dynamic':
            perturb_updatelist, w, adv_vec = multi_gradnorm(model, dataset_central, oris,
                                                                                  full_data, hyper)
        elif method == 'uni-const':
            print('check method should be uni equal', method)
            perturb_updatelist, adv_vec = multi(model, dataset_central, oris, full_data, hyper)
        elif method == 'deepbillboard':
            perturb_updatelist, adv_vec = deepbillboard(model, dataset_central, oris, full_data, hyper)
        else:
            raise Exception('Error: No method was given!')

        s_adv, t_adv = adv_vec[-1][0], adv_vec[-1][1]
        serror, terror = (s_adv-ori_steer_pred) * d[0], (t_adv-ori_throttle_pred) * d[1]
        compute_metric(hyper, serror, terror)
        success_rate(serror, terror, hyper)
        total_sadv.append(s_adv)
        total_tadv.append(t_adv)
    mean_sadv = np.mean(total_sadv, axis=0)
    mean_tadv = np.mean(total_tadv, axis=0)

    ori_throttle_pred, ori_steer_pred = ori_output[:, 1].data.cpu().numpy(), ori_output[:,0].data.cpu().numpy()
    mean_serror, mean_terror = (mean_sadv - ori_steer_pred) * d[0], (mean_tadv - ori_throttle_pred)*d[1]
    return mean_serror, mean_terror



start = time.time()
if method =='uni-dynamic' or method=='uni-const':
    hyper = {}
    hyper['method'] = method
    hyper['direction'] = args.direction
    print('hyper direction', hyper['direction'])
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
    if method == 'uni-const':
        if args.weights == 'equal':
            hyper['weights'] = [1/3, 1/3, 1/3]
        elif args.weights == 'steer':
            hyper['weights'] = [1, 0, 0]
        elif args.weights == 'throttle':
            hyper['weights'] = [0, 1, 0]
        else:
            raise Exception('Error! Wrong args.weights given!')
    mean_serror, mean_terror = training_uni(hyper, model, full_dataset, augmenter)
    success_rate(mean_serror, mean_terror, hyper)
    compute_metric(hyper, mean_serror, mean_terror)

if method=='FGSM':
    hyper2 = {}
    hyper2['method'] = method
    hyper2['weight_strategy'] = args.weights  # 'equal' or 'grid search best'
    if hyper2['weight_strategy'] == 'grid search best':
        hyper2['times'] = args.times   #grid search times
    hyper2['eps'] = args.eps
    hyper2['lr'] = args.lr
    hyper2['direction'] = args.direction
    hyper2['beta'] = args.beta
    hyper2['iters'] = args.iters
    hyper2['title'] = args.title   #title of the video data

    serror_framelist, terror_framelist = run_fgsm(hyper2, model, full_dataset, augmenter)
    success_rate(np.array(serror_framelist), np.array(terror_framelist), hyper2)
    compute_metric(hyper2, serror_framelist, terror_framelist)




'''
run deep billboard
'''

def run_deepbillboard(hyper):
    create_folder(hyper['method'], hyper['title'])
    mean_serror, mean_terror = training_uni(hyper, model, full_dataset, augmenter)
    print('final average over {0} runs metric:'.format(hyper['runs']))
    success_rate(mean_serror, mean_terror, hyper)
    compute_metric(hyper, mean_serror, mean_terror)

if method == 'deepbillboard':
    tuned_hyper = {}
    tuned_hyper['zip3,00715,white_car,raining'] = [42800,0.025]
    tuned_hyper['zip6_epi02431_black car'] = [42800, 0.5]
    tuned_hyper['zip06,car,epi02472'] = [42800,0.5]
    tuned_hyper['zip3,00714,blue_car'] = [1000,0.025]
    tuned_hyper['zip14,05012,red_light'] = [22800, 0.5]
    tuned_hyper['episode_00000_pedestrian'] = [42800,0.25]
    tuned_hyper['zip14,05013,red_light'] = [42800,0.25]

    hyper3 = {}
    hyper3['iters'] = 250
    hyper3['jsma'] = True
    hyper3['strategy'] = args.strategy
    hyper3['direction'] = args.direction
    hyper3['bs'] = 5
    hyper3['simulated_annealing'] = True
    hyper3['eps'] = 2.0
    hyper3['sa_k'] = 30
    hyper3['sa_b'] = 0.96
    hyper3['runs'] = args.runs
    #hyper3['lr'] = args.lr
    hyper3['method'] = 'deepbillboard'

    hyper3['title'] = args.title
    hyper3['jsma_n'] = tuned_hyper[args.title][0]
    hyper3['lr'] = tuned_hyper[args.title][1]
    run_deepbillboard(hyper3)





