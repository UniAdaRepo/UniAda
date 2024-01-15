'''
DeepBillboard
'''
import torch
import numpy as np
import copy
import random
import math

def find_kth_max(array, k):
    tmp = array.flatten()
    tmp = abs(tmp)
    tmp.sort()
    return tmp[-k]


def deepbillboard(model, dataset_central, oris, full_data, hyper):
    print("check hyper['direction']", hyper['direction'])
    d = hyper['direction']
    print('check d separately', d[0], d[1], d[2])
    strategy, eps, lr, bs, iterations, ds = hyper['strategy'], hyper['eps'], hyper['lr'], hyper['bs'], hyper['iters'], hyper['direction'][0]
     #only retrieve steer direction ds
    print('ds', ds, 'eps', eps)
    ori_output, ori_speed, ori_controls = oris['output'], oris['speed'], oris['controls']
    ori_throttle_pred, ori_brake_pred, ori_steer_pred = ori_output[:, 1], ori_output[:, 2], ori_output[:, 0]
    total_perturb = torch.zeros((3, 88, 200))
    imgs_ori = copy.deepcopy(full_data['rgb'])  # all imgs_ori
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(full_data['rgb'])
    real_total_perturb_iterlist, adversarial = [], []
    last_diff = 0
    print('k', hyper['jsma_n'], 'data', hyper['title'], 'lr', hyper['lr'])

    for i in range(iterations):
        # print('iter',i)
        if i % 50 == 0 and i != 0:
            lr = lr * 0.8
        torch.cuda.empty_cache()
        np.random.shuffle(idx)  # shuffle at every iteration
        for j in range(0, len(imgs), bs):  # for each minibatch
            torch.cuda.empty_cache()
            num = 0
            minibatch = {}
            if j <= len(imgs) - bs:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, j + bs)]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, j + bs)]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, j + bs)]
                perturb = torch.zeros((bs, 3, 88, 200))

            else:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, len(imgs))]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, len(imgs))]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, len(imgs))]
                perturb = torch.zeros((len(imgs) - j, 3, 88, 200))

            # for each sample
            for image, speed, controls in zip(minibatch['rgb'], minibatch['speed_module'],
                                              minibatch['directions']):  # for a single image in mini-batch
                image.requires_grad = True

                o, _ = model.forward_branch(image[None, ...].cuda(), speed[None, ...].cuda(), controls[None, ...])
                output = o[0]

                loss = -output[0] * ds # minimize the loss, ds=1 to steer right (max steer)
                loss.backward()  # single image loss
                grads = image.grad.data.clone()
                # print('grads before jsma', grads)
                if hyper['jsma']:  # we only consider top k grad pixels
                    k_th_value = find_kth_max(grads, hyper['jsma_n'])
                    super_threshold_indices = abs(grads) < k_th_value
                    grads[super_threshold_indices] = 0
                perturb[num] = grads
                # print('current grads', grads)
                num += 1
                image.grad.data.zero_()

            # now combine batch-size perturbation proposals into a single
            if strategy == 'max':  # max or sum, not mean
                # print('max strategy')
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]
            if strategy == 'sum':
                combine = torch.sum(perturb, dim=0)

            tmp_total_perturb = torch.clamp(total_perturb - lr * combine, min=-eps/255, max=eps/255)  
            tmp_adv_imgs = torch.clamp(torch.add(imgs_ori, tmp_total_perturb[None, ...]), min=0,
                                       max=1)  # update ALL images (not only those in minibatch)

            # tmp_adv_imgs need to be in the same form as data['rgb']
            with torch.no_grad():
                output, _ = model.forward_branch(tmp_adv_imgs.cuda(),
                                                 dataset_central.extract_inputs(full_data).cuda(),
                                                 full_data['directions'])
                adv_steer_vec, adv_throttle_vec, adv_brake_vec = output[:, 0], output[:, 1], output[:,
                                                                                             2] 

            tmp_error_s = np.sum(np.abs(adv_steer_vec.data.cpu().numpy() - ori_steer_pred.data.cpu().numpy()))
            this_diff = tmp_error_s
            if this_diff > last_diff:  # if satisfied, we update
                # print('good update')
                total_perturb = copy.deepcopy(tmp_total_perturb)
                last_diff = this_diff
                imgs = copy.deepcopy(tmp_adv_imgs.cpu())
                real_total_perturb_iterlist.append(total_perturb)
                adversarial.append([adv_steer_vec.data.cpu().numpy(), adv_throttle_vec.data.cpu().numpy(),
                                    adv_brake_vec.data.cpu().numpy()])
            else:
                if hyper['simulated_annealing']:
                    if (random.random() < pow(math.e, hyper['sa_k'] * (this_diff - last_diff) / (
                    pow(hyper['sa_b'], i))) and this_diff != last_diff):
                        total_perturb = copy.deepcopy(tmp_total_perturb)
                        imgs = copy.deepcopy(tmp_adv_imgs.cpu())
                        last_diff = this_diff
                        # print('bad update')
                        real_total_perturb_iterlist.append(total_perturb)
                        adversarial.append([adv_steer_vec.data.cpu().numpy(), adv_throttle_vec.data.cpu().numpy(),
                                            adv_brake_vec.data.cpu().numpy()])

    return real_total_perturb_iterlist, adversarial



def deepbillboard_multi(model, dataset_central, oris, full_data, hyper):
    print('enter deepbillboard_multi equal wegiths')
    print("check hyper['direction']", hyper['direction'])
    d = hyper['direction']
    print('check d separately', d[0], d[1], d[2])
    strategy, eps, lr, bs, iterations, ds = hyper['strategy'], hyper['eps'], hyper['lr'], hyper['bs'], hyper['iters'], hyper['direction'][0]
    beta = hyper['beta']

    ori_output, ori_speed, ori_controls = oris['output'], oris['speed'], oris['controls']
    ori_throttle_pred, ori_brake_pred, ori_steer_pred = ori_output[:, 1], ori_output[:, 2], ori_output[:, 0]
    total_perturb = torch.zeros((3, 88, 200))
    imgs_ori = copy.deepcopy(full_data['rgb'])  # all imgs_ori
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(full_data['rgb'])
    real_total_perturb_iterlist, adversarial = [], []
    last_diff_s, last_diff_t, last_diff_b = 0, 0, 0
    print('k', hyper['jsma_n'], 'data', hyper['title'], 'lr', hyper['lr'])

    for i in range(iterations):
        # print('iter',i)
        if i % 50 == 0 and i != 0:
            lr = lr * 0.8
        torch.cuda.empty_cache()
        np.random.shuffle(idx)  # shuffle at every iteration
        for j in range(0, len(imgs), bs):  # for each minibatch
            torch.cuda.empty_cache()
            num = 0
            minibatch = {}
            if j <= len(imgs) - bs:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, j + bs)]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, j + bs)]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, j + bs)]
                perturb = torch.zeros((bs, 3, 88, 200))

            else:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, len(imgs))]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, len(imgs))]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, len(imgs))]
                perturb = torch.zeros((len(imgs) - j, 3, 88, 200))

            # for each sample
            for image, speed, controls in zip(minibatch['rgb'], minibatch['speed_module'],
                                              minibatch['directions']):  # for a single image in mini-batch
                image.requires_grad = True

                o, _ = model.forward_branch(image[None, ...].cuda(), speed[None, ...].cuda(), controls[None, ...])
                output = o[0]

                #loss = -output[0] * ds # minimize the loss, ds=1 to steer right (max steer)
                lossA, lossB, lossC = 1 / beta * torch.exp(output[0] * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                    output[1] * (-1 / beta) * d[1]), 1 / beta * torch.exp(output[2] * (-1 / beta) * d[2])
                loss = 1/3. * lossA + 1./3 * lossB + 1./3 * lossC
                loss.backward()  # single image loss
                grads = image.grad.data.clone()
                # print('grads before jsma', grads)
                if hyper['jsma']:  # we only consider top k grad pixels
                    k_th_value = find_kth_max(grads, hyper['jsma_n'])
                    super_threshold_indices = abs(grads) < k_th_value
                    grads[super_threshold_indices] = 0
                perturb[num] = grads
                # print('current grads', grads)
                num += 1
                image.grad.data.zero_()

            # now combine batch-size perturbation proposals into a single
            if strategy == 'max':  # max or sum, not mean
                # print('max strategy')
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]
            if strategy == 'sum':
                combine = torch.sum(perturb, dim=0)

            tmp_total_perturb = torch.clamp(total_perturb - lr * combine, min=-eps/255, max=eps/255)  # of size (3,88,200)
            tmp_adv_imgs = torch.clamp(torch.add(imgs_ori, tmp_total_perturb[None, ...]), min=0,
                                       max=1)  # update ALL images (not only those in minibatch)

            # tmp_adv_imgs need to be in the same form as data['rgb']
            with torch.no_grad():
                output, _ = model.forward_branch(tmp_adv_imgs.cuda(),
                                                 dataset_central.extract_inputs(full_data).cuda(),
                                                 full_data['directions'])
                adv_steer_vec, adv_throttle_vec, adv_brake_vec = output[:, 0], output[:, 1], output[:,
                                                                                             2]  # current throttle pred

            tmp_error_s = np.sum(np.abs(adv_steer_vec.data.cpu().numpy() - ori_steer_pred.data.cpu().numpy()))
            tmp_error_t = np.sum(np.abs(adv_throttle_vec.data.cpu().numpy() - ori_throttle_pred.data.cpu().numpy()))
            tmp_error_b = np.sum(np.abs(adv_brake_vec.data.cpu().numpy() - ori_brake_pred.data.cpu().numpy()))
            this_diff_s, this_diff_t, this_diff_b = tmp_error_s, tmp_error_t, tmp_error_b
            if (this_diff_s > last_diff_s) or (this_diff_t > last_diff_t) or (this_diff_b > last_diff_b):  # if satisfied, we update
                # print('good update')
                total_perturb = copy.deepcopy(tmp_total_perturb)
                last_diff_s, last_diff_t, last_diff_b = this_diff_s, this_diff_t, this_diff_b
                imgs = copy.deepcopy(tmp_adv_imgs.cpu())
                real_total_perturb_iterlist.append(total_perturb)
                adversarial.append([adv_steer_vec.data.cpu().numpy(), adv_throttle_vec.data.cpu().numpy(),
                                    adv_brake_vec.data.cpu().numpy()])
            else:
                if hyper['simulated_annealing']:
                    if (random.random() < pow(math.e, hyper['sa_k'] * (this_diff_s - last_diff_s) / (
                    pow(hyper['sa_b'], i))) and this_diff_s != last_diff_s):
                        total_perturb = copy.deepcopy(tmp_total_perturb)
                        imgs = copy.deepcopy(tmp_adv_imgs.cpu())
                        last_diff_s, last_diff_t, last_diff_b = this_diff_s, this_diff_t, this_diff_b
                        # print('bad update')
                        real_total_perturb_iterlist.append(total_perturb)
                        adversarial.append([adv_steer_vec.data.cpu().numpy(), adv_throttle_vec.data.cpu().numpy(),
                                            adv_brake_vec.data.cpu().numpy()])

    return real_total_perturb_iterlist, adversarial
