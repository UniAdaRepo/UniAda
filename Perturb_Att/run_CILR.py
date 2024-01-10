
import sys
sys.path.append('./gradnorm/')
import os
from configs import g_conf, merge_with_yaml
from input import Augmenter, coil_valid_dataset
import torch
import numpy as np
from network import CoILModel
import copy
import logging
from torch.autograd import Variable

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


def success_rate(mean_serror, mean_terror, logger):
    logger.info('--------------success rate in % on average----------------')
    s_threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    t_threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.748, 0.8, 0.9]

    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} '.format(np.round(srate, 4), s * 70))


    for t in t_threshold:
        trate = np.sum(mean_terror > t) / len(mean_terror) * 100
        logger.info('{0}% of frames have throttle error > {1} '.format(np.round(trate, 4), t * 46))


'''
compute mean error and median error metrics
'''
iters = 4
alpha = 1/255
def compute_metric(serror_framelist, terror_framelist, logger):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(title, method, iters))
    ME_s = np.mean(serror_framelist)
    ME_t = np.mean(terror_framelist)
    logger.info('s={0}, t={1}'.format(ME_s * 70, ME_t * 46))

    logger.info('-----median error {0} for {1}-----'.format(title, method))
    median_s = np.median(serror_framelist)
    median_t = np.median(terror_framelist)
    logger.info('s={0}, t={1}'.format(median_s * 70, median_t * 46))


def pgd(logger, eps=2/255, iter_num=4, alpha=1/255, beta=2.0):
    logger.info('iter_num={0}, alpha={1}'.format(iter_num, alpha))
    d = [1,1,-1]

    data_loader = torch.utils.data.DataLoader(dataset_central, batch_size=1,
                                              shuffle=False)
    s_adv, t_adv, b_adv, etas = [], [], [], []
    for i, data in enumerate(data_loader):
        imgs_ori = torch.squeeze(data['rgb'])
        image = Variable(imgs_ori, requires_grad=True)
        for iteration in range(iter_num):
            o, _ = model.forward_branch(image[None,...].cuda(), dataset_central.extract_inputs(data).cuda(),
                                        data['directions'])
            output = o[0]
            adv_throttle, adv_steer = output[1], output[0]

            lossA, lossB, lossC = 1 / beta * torch.exp(output[0] * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                output[1] * (-1 / beta) * d[1]), 1 / beta * torch.exp(output[2] * (-1 / beta) * d[2])
            loss = 1/3 * lossA + 1/3 * lossB + 1/3 * lossC
            loss.backward(retain_graph=True)
            # pgd attack
            adv_img_data = image.data - alpha * torch.sign(image.grad.data)
            eta = torch.clamp(adv_img_data - imgs_ori, min=-eps, max=eps)
            image.data = torch.clamp(imgs_ori + eta, min=0, max=1)
            image.grad.data.zero_()

        etas.append(eta.data.cpu().numpy())
        s_adv.append(
            adv_steer.data.cpu().numpy())
        t_adv.append(adv_throttle.data.cpu().numpy())
    return s_adv, t_adv, etas
method = 'Perturb_att'
# load CILR model
os.environ["COIL_DATASET_PATH"] = './UniAda-main'
augmenter = Augmenter(None)
merge_with_yaml(os.path.join('./gradnorm/configs/nocrash/resnet34imnet10-nospeed.yaml'))
model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
checkpoint = torch.load(os.path.join('./gradnorm/_logs/nocrash/resnet34imnet10-nospeed/checkpoints/760000.pth'))
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()  # this is for turn off dropout
full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
titles = ['episode_00000_pedestrian', 'zip3,00715,white_car,raining', 'zip6_epi02431_black car',
          'zip06,car,epi02472', 'zip3,00714,blue_car',
          'zip14,05013,red_light', 'OK,zip14,epi04600,car']
save_path = '/home/jzhang2297/gradnorm/results/CILR_logger/Perturb_att/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for run in range(1):
    for title in titles:
        print(title)
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
        imgs = copy.deepcopy(full_data['rgb'])
        logger = set_logger(os.path.join(save_path, 'CILR_{0}_{1}.log'.format(method, title)), title, run)
        logger.info('method = {0}, direction = {1}, video = {2}, eps={3}'.format(method, [1, 1, -1], title, 2))
        s_adv, t_adv, etas = pgd(logger=logger, iter_num=iters, alpha=alpha)

        np.save('PA_CILR_perturb_{0}_iters{1}.npy'.format(title, iters), etas)

        serror = (np.array(s_adv)-ori_steer_pred) * 1
        terror = (np.array(t_adv)-ori_throttle_pred) * 1
        logger.info('compute single run metric')
        compute_metric(serror, terror, logger)
        success_rate(serror, terror, logger)
