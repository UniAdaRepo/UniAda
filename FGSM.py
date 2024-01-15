import numpy as np
import os
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from input import coil_valid_dataset
from torch.autograd import Variable

'''
Perform FGSM (iters=2)
'''


def fgsmattack(model, data, dataset_central, hyper, w):
    eps, iterations, beta, lr, d = hyper['eps'], hyper['iters'], hyper['beta'], hyper['lr'], hyper['direction']
    w = torch.tensor(w).type(torch.FloatTensor).cuda()
    steer_iterlist, throttle_iterlist, brake_iterlist, steer_iterlist['adv'], throttle_iterlist['adv'], brake_iterlist[
        'adv'], real_total_perturb_iterlist = {}, {}, {}, [], [], [], []

    imgs_ori = torch.squeeze(data['rgb'])
    image = Variable(imgs_ori, requires_grad=True)
    steer_iterlist['gt'], throttle_iterlist['gt'], brake_iterlist['gt'], controls = data['steer'], data['throttle'], \
                                                                                    data['brake'], data['directions']

    for i in range(iterations):
        o, _ = model.forward_branch(image[None, ...].cuda(), dataset_central.extract_inputs(data).cuda(),
                                    controls)  # extract steer+throttle+brake
        output = o[0]
        adv_throttle, adv_steer, adv_brake = output[1], output[0], output[2]

        steer_iterlist['adv'].append(adv_steer.data.cpu().numpy())  # note that first one is the original steer&throttle
        throttle_iterlist['adv'].append(adv_throttle.data.cpu().numpy())
        brake_iterlist['adv'].append(adv_brake.data.cpu().numpy())
        real_total_perturb_iterlist.append(image.data - imgs_ori)

        lossA, lossB, lossC = 1 / beta * torch.exp(output[0] * (-1 / beta) * d[0]), 1 / beta * torch.exp(
            output[1] * (-1 / beta) * d[1]), 1 / beta * torch.exp(output[2] * (-1 / beta) * d[2])
        loss = w[0] * lossA + w[1] * lossB + w[2] * lossC
        loss.backward()
        perturbation = lr * torch.sign(image.grad.data)

        perturbation = torch.clamp((image.data - perturbation) - imgs_ori, min=-eps / 255,
                                   max=eps / 255)  # if clip, alpha needs < 1/255

        image.data = torch.clamp(imgs_ori + perturbation, min=0, max=1)

        image.grad.data.zero_()  # flush gradients

        # adaptive lr:
        if i % 50 == 0 and i != 0:
            lr = lr * 0.8

    steer_iterlist['ori_pred'], throttle_iterlist['ori_pred'], brake_iterlist['ori_pred'] = steer_iterlist['adv'][0], \
                                                                                            throttle_iterlist['adv'][0], \
                                                                                            brake_iterlist['adv'][0]
    return steer_iterlist, throttle_iterlist, brake_iterlist, real_total_perturb_iterlist



def grid_search(hyper, model, full_dataset, augmenter):
    d = hyper['direction']
    record, ws = [], []
    print('title for fgsm grid search', hyper['title'])
    dataset_central = coil_valid_dataset.CoILDataset_central_valid(full_dataset, hyper['title'], transform=augmenter,
                                                                   preload_name=None)

    data_loader = torch.utils.data.DataLoader(dataset_central, batch_size=1,
                                              shuffle=False)  # batch size=1 is hard-coded for fgsm/BIM
    for data in data_loader:
        record, weights = [], []
        for search in np.arange(hyper['times']):  # retrieve random weights 100 times
            r = np.random.uniform(size=3)  # this is hard-coded, we have 3 tasks. min=0, max=1
            w = r / np.sum(r)  # resize so sum to 1
            s_iterlist, t_iterlist, b_iterlist, _ = fgsmattack(model, data, dataset_central, hyper, w)
            t_error = (t_iterlist['adv'][-1] - t_iterlist['ori_pred']) * d[1]  # take 2nd one, hardcoded, scalar
            s_error = (s_iterlist['adv'][-1] - s_iterlist['ori_pred']) * d[0]
            b_error = (b_iterlist['adv'][-1] - b_iterlist['ori_pred']) * d[2]

            if b_error > 0 and t_error > 0 and s_error > 0:  # only record if w meets correct attack direction for all s,t,b
                metric = np.mean([s_error, t_error, b_error])
                weights.append(w)
                record.append(metric)


        if weights == []:  # if no weight samples meet correct attack direction
            best_weight = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            print('best weight index', np.argmax(np.array(record)))
            best_weight = weights[np.argmax(np.array(record))]
        ws.append(best_weight)
    return ws  


def run_fgsm(hyper, model, full_dataset, augmenter):
    d = hyper['direction']
    dataset_central = coil_valid_dataset.CoILDataset_central_valid(full_dataset, hyper['title'], transform=augmenter,
                                                                   preload_name=None)

    data_loader = torch.utils.data.DataLoader(dataset_central, batch_size=1,
                                              shuffle=False)  # batch size=1 is hard-coded for fgsm/BIM
    serror_framelist, terror_framelist, berror_framelist = [], [], []
    if hyper['weight_strategy'] == 'equal':
        w = [1 / 3, 1 / 3, 1 / 3]

    if hyper['weight_strategy'] == 'grid search best':
        ws = grid_search(hyper, model, full_dataset, augmenter)
        
    for i, data in enumerate(data_loader):  # load a single image frame
        if hyper['weight_strategy'] == 'grid search best':
            print('grid search best weight')
            w = ws[i]
        s_iterlist, t_iterlist, b_iterlist, perturb_iter = fgsmattack(model, data, dataset_central, hyper, w)

        t_error = (t_iterlist['adv'][-1] - t_iterlist['ori_pred']) * d[1]  # take 2nd one, hardcoded
        s_error = (s_iterlist['adv'][-1] - s_iterlist['ori_pred']) * d[0]
        b_error = (b_iterlist['adv'][-1] - b_iterlist['ori_pred']) * d[2]
        serror_framelist.append(s_error)
        terror_framelist.append(t_error)
        berror_framelist.append(b_error)
    return serror_framelist, terror_framelist, berror_framelist
