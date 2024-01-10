

import numpy as np
import torch
import copy
from torch.autograd import Variable

from plot_log import *


def multi(model, dataset_central, oris, full_data, hyper):
    print('enter multi')
    eps, lr, bs, beta, iterations, norm_threshold, d, e = hyper['eps'], hyper['lr'], hyper['bs'], hyper['beta'],\
                                                       hyper['iters'], hyper['norm_threshold'], hyper['direction'], hyper['weights']
    print('weights', e)
    print('eps', eps)
    print('directions', d)
    ori_output, ori_speed, ori_controls = oris['output'], oris['speed'], oris['controls']
    total_perturb = torch.zeros((3, 88, 200))
    imgs_ori = copy.deepcopy(full_data['rgb'])     #all imgs_ori
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(full_data['rgb'])
    total_perturb_iterlist, adversarial = [],[]
   
    for i in range(iterations):
        if i%50==0 and i != 0:
            lr = lr*0.8
        torch.cuda.empty_cache()
        np.random.shuffle(idx)       #shuffle at every iteration
        for j in range(0,len(imgs),bs):   #for each minibatch
            torch.cuda.empty_cache()
            num=0
            minibatch={}
            if j <= len(imgs) - bs:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j,j+bs)]
                minibatch['speed_module']=[ori_speed[idx[k]] for k in range(j,j+bs)]
                minibatch['directions']=[ori_controls[idx[k]] for k in range(j,j+bs)]
                perturb = torch.zeros((bs, 3, 88, 200))
             
            else:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j,len(imgs))]
                minibatch['speed_module']=[ori_speed[idx[k]] for k in range(j,len(imgs))]
                minibatch['directions']=[ori_controls[idx[k]] for k in range(j,len(imgs))]
                perturb = torch.zeros((len(imgs)-j, 3, 88, 200))
       
            #for each sample
            for image, speed, controls in zip(minibatch['rgb'], minibatch['speed_module'], minibatch['directions']):#for a single image in mini-batch
                image.requires_grad=True
                
                o,_ = model.forward_branch(image[None,...].cuda(), speed[None,...].cuda(), controls[None,...])
                output=o[0]
                
                #now minimize the loss
                lossA, lossB, lossC = 1/beta*torch.exp(output[0]*(-1/beta)*d[0]), 1/beta*torch.exp(output[1]*(-1/beta)*d[1]), \
                                                                                  1/beta*torch.exp(output[2]*(-1/beta)*d[2])

                loss = e[0]*lossA+e[1]*lossB+e[2]*lossC   #minimize the loss, equal weights, sum to 1
                loss.backward()
                perturb[num] = image.grad.data
                num += 1
                image.grad.data.zero_()
                
        #now combine batch-size perturbation proposals into a single
            if hyper['strategy'] == 'mean':
                combine = torch.mean(perturb, dim=0)

            if hyper['strategy'] == 'max':
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]

            if torch.norm(combine) < norm_threshold and torch.norm(combine)!=0:
                combine = combine * norm_threshold / torch.norm(combine)
            
            
            total_perturb = torch.clamp(total_perturb - lr*combine, min=-eps/255, max=eps/255)
            adv_imgs = torch.clamp(torch.add(imgs_ori,total_perturb[None,...]), min=0, max=1)
            

            with torch.no_grad():
                output,_ = model.forward_branch(adv_imgs.cuda(), 
                                            dataset_central.extract_inputs(full_data).cuda(), full_data['directions'])
                adv_steer_vec, adv_throttle_vec = output[:,0], output[:,1]
     
    #update all images 
            imgs = copy.deepcopy(adv_imgs.cpu())
            total_perturb_iterlist.append(total_perturb)
            adversarial.append([adv_steer_vec.data.cpu().numpy(),adv_throttle_vec.data.cpu().numpy()])
            
    return total_perturb_iterlist, adversarial


def multi_gradnorm(model, dataset_central, oris, full_data, hyper):
    eps, lr, bs, beta, alpha, lrgrad, iterations, norm_threshold, d = hyper['eps'], hyper['lr'], hyper['bs'], hyper[
        'beta'], hyper['alpha'], hyper['lrgrad'], hyper['iters'], hyper['norm_threshold'], hyper['direction']
    ori_output, ori_speed, ori_controls = oris['output'], oris['speed'], oris['controls']
    ori_throttle_pred, ori_brake_pred, ori_steer_pred = ori_output[:, 1], ori_output[:, 2], ori_output[:, 0]
    init_loss_s = torch.mean(1 / beta * torch.exp(ori_steer_pred * (-1 / beta) * d[0]))  # avg over all imgs
    init_loss_t = torch.mean(1 / beta * torch.exp(ori_throttle_pred * (-1 / beta) * d[1]))
    init_loss_b = torch.mean(1 / beta * torch.exp(ori_brake_pred * (-1 / beta) * d[2]))
    initial_task_loss = torch.tensor([init_loss_s, init_loss_t, init_loss_b])
    total_perturb = torch.zeros((3, 88, 200))
    imgs_ori = copy.deepcopy(full_data['rgb'])  # all original images in this video
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(full_data['rgb'])
    real_total_perturb_iterlist, adversarial = [], []
    a1, a2, a3 = Variable(torch.tensor(1/3), requires_grad=True).cuda(), Variable(torch.tensor(1/3),
                                                                                    requires_grad=True).cuda(), Variable(
        torch.tensor(1/3), requires_grad=True).cuda()
    for i in range(iterations):
        if i % 50 == 0 and i != 0:
            lr = lr * 0.8
            lrgrad = lrgrad * 0.8
        torch.cuda.empty_cache()
        np.random.shuffle(idx)  # shuffle at every iteration
        for j in range(0, len(imgs), bs):  # for each minibatch
            torch.cuda.empty_cache()
            num = 0
            minibatch = {}
            if j <= len(imgs) - bs:  # create mini-batch
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, j + bs)]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, j + bs)]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, j + bs)]
                perturb = torch.zeros((bs, 3, 88, 200))

            else:  # create mini-batch
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, len(imgs))]
                minibatch['speed_module'] = [ori_speed[idx[k]] for k in range(j, len(imgs))]
                minibatch['directions'] = [ori_controls[idx[k]] for k in range(j, len(imgs))]
                perturb = torch.zeros((len(imgs) - j, 3, 88, 200))

            total_norms = torch.tensor([[0.0, 0.0, 0.0]]).cuda()
            unweighted_losses = []

            # for each data sample
            for image, speed, controls in zip(minibatch['rgb'], minibatch['speed_module'],
                                              minibatch['directions']):  # for a single image in mini-batch
                image.requires_grad = True
                o, _ = model.forward_branch(image[None, ...].cuda(), speed[None, ...].cuda(), controls[None, ...])
                output = o[0]

                # now minimize the loss
                lossA, lossB, lossC = 1 / beta * torch.exp(output[0] * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                    output[1] * (-1 / beta)*d[1]), 1 / beta * torch.exp(output[2] * (-1 / beta)*d[2])
                # max the steer(steer right) to min lossA, max the throttle to min lossB, min brake to min lossC

                loss = a1 * lossA + a2 * lossB + a3 * lossC  # minimize the loss
                loss.backward(retain_graph=True)  # single image loss
                unweighted_losses.append([lossA.data.cpu(), lossB.data.cpu(), lossC.data.cpu()])

                perturb[num] = image.grad.data
                num = num + 1

                single_norm = torch.tensor([0.0]).cuda()

                for t, a in zip([lossA, lossB, lossC], [a1, a2, a3]):
                    gygw = torch.autograd.grad(t, image, retain_graph=True)
                    cop = torch.unsqueeze(torch.norm(torch.mul(a, gygw[0].cuda())), dim=0)

                    single_norm = torch.cat((single_norm, cop), 0)

                single_norm = single_norm[1:]
                total_norms = torch.cat((total_norms, torch.unsqueeze(single_norm, dim=0)), 0)
                image.grad.data.zero_()

            total_norms = total_norms[1:]
            norms = torch.mean(total_norms, dim=0)  # average over all batch images size 1*3

            task_loss = torch.mean(torch.tensor(unweighted_losses), dim=0)  # average over all batch imgs' losses

            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss

            inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
            mean_norm = np.mean(norms.data.cpu().numpy())
            constant_term = (mean_norm * (inverse_train_rate ** alpha)).clone().detach().cuda()

            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
            a1.grad = torch.autograd.grad(grad_norm_loss, a1, retain_graph=True)[0]
            a2.grad = torch.autograd.grad(grad_norm_loss, a2, retain_graph=True)[0]
            a3.grad = torch.autograd.grad(grad_norm_loss, a3, retain_graph=True)[0]

            tmp_a1, tmp_a2, tmp_a3 = a1 - lrgrad * a1.grad, a2 - lrgrad * a2.grad, a3 - lrgrad * a3.grad
            if tmp_a1 > 0:  # update only if new a_i is positive
                a1 = tmp_a1
            if tmp_a2 > 0:
                a2 = tmp_a2
            if tmp_a3 > 0:
                a3 = tmp_a3
            # renormalize sum to 1
            normalize_coeff = 1 / (a1 + a2 + a3)
            a1, a2, a3 = a1 * normalize_coeff, a2 * normalize_coeff, a3 * normalize_coeff

            if hyper['strategy'] == 'mean':
                combine = torch.mean(perturb, dim=0)

            if hyper['strategy'] == 'max':
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]

            if torch.norm(combine) < norm_threshold and torch.norm(combine) != 0:
                combine = combine * norm_threshold / torch.norm(combine)

            total_perturb = torch.clamp(total_perturb - lr * combine, min=-eps/255, max=eps/255)
            adv_imgs = torch.clamp(torch.add(imgs_ori, total_perturb[None, ...]), min=0, max=1)

            with torch.no_grad():
                output, _ = model.forward_branch(adv_imgs.cuda(),
                                                 dataset_central.extract_inputs(full_data).cuda(),
                                                 full_data['directions'])
                adv_steer_vec, adv_throttle_vec = output[:, 0], output[:, 1]

            imgs = copy.deepcopy(adv_imgs.cpu())
            real_total_perturb_iterlist.append(total_perturb)
            adversarial.append([adv_steer_vec.data.cpu().numpy(), adv_throttle_vec.data.cpu().numpy()])

    return real_total_perturb_iterlist, adversarial








