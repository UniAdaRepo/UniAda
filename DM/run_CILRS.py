from main import DeepManeuver
import sys
sys.path.append('./gradnorm/')
import os
from configs import g_conf, merge_with_yaml
from input import Augmenter, coil_valid_dataset
from network import CoILModel
import torch
import numpy as np
from network import CoILModel
import copy
import logging

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

def success_rate(mean_serror, logger, title):
    logger.info('--------------success rate in % on average----------------')
    s_threshold = [0.05, 0.2, 0.3, 0.4]

    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} degrees '.format(np.round(srate, 4), s*70))


'''
compute mean error and median error metrics
'''
def compute_metric(serror_framelist, logger, title):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(title, 'DeepManeuver', 400))
    ME_s = np.mean(serror_framelist)
    logger.info('ME_s={0} degrees'.format(ME_s * 70))

    logger.info('-----median error {0} for {1}-----'.format(title, 'DeepManeuver'))
    median_s = np.median(serror_framelist)
    logger.info('Med_s={0} degrees'.format(median_s * 70))

def deepmaneuver(model, img_arr, direction, steering_vector, bb_size=5, iterations=400, noise_level=25,
                       dist_to_bb=None, last_billboard=None, input_divers=True, loss_fxn='inv23'):
    sdbb = DeepManeuver(model, None, direction)
    tensorized_steering_vector = torch.as_tensor(steering_vector, dtype=torch.float)
    y, perturb = sdbb.perturb_images({'imgs':img_arr, 'speed':ori_speed, 'controls':ori_controls}, model,
                                                        tensorized_steering_vector, bb_size=bb_size,
                                                        iterations=iterations, noise_level=noise_level,
                                                        last_billboard=last_billboard, loss_fxn=loss_fxn, input_divers=input_divers)
    return y, perturb
# load CILRS model
os.environ["COIL_DATASET_PATH"] = './UniAda-main'
augmenter = Augmenter(None)
merge_with_yaml(os.path.join('./gradnorm/configs/nocrash/resnet34imnet10S1.yaml'))
model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
checkpoint = torch.load(os.path.join('./gradnorm/_logs/nocrash/resnet34imnet10S1/checkpoints/660000.pth'))
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()  # this is for turn off dropout
full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
titles = ['episode_00000_pedestrian', 'zip3,00715,white_car,raining', 'zip6_epi02431_black car',
          'zip06,gray_car,epi02472', 'zip3,00714,blue_car',
          'zip14,05013,red_light', 'zip14,epi04600,light_blue_car']
save_path = './gradnorm/results/CILRS_logger/DeepManeuver/'
noises = [15]
iter=400
for noise in noises:
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
        ori_steer_pred = ori_output[:,0].data.cpu().numpy()
        imgs = copy.deepcopy(full_data['rgb'])

        sequence2 = imgs
        direction = 'right'
        steering_vector2 = np.ones(len(imgs))
        s_adv, perturb = deepmaneuver(model, sequence2, direction, copy.deepcopy(steering_vector2),
                                                           bb_size=5, iterations=iter, noise_level=noise,
                                                           dist_to_bb=1, input_divers=False, loss_fxn="MDirE")

        np.save('DM_CILRS_perturb_{0}_noise{1}.npy'.format(title, noise), perturb.data.cpu().numpy())
        logger = set_logger(os.path.join(save_path, 'CILRS_{0}_{1}.log'.format('DeepManeuver', title)), title, noise)
        logger.info('method = {0}, direction = {1}, video = {2}, eps={3}, noise={4}'.format('DeepManeuver', 'steering=right', title, 2, noise))
        serror = (s_adv.data.cpu().numpy()-ori_steer_pred) * 1
        logger.info('compute single run metric')
        compute_metric(serror, logger, title)
        success_rate(serror, logger, title)
