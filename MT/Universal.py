

import numpy as np
import os
import torch
import copy
from torch.autograd import Variable
from torchvision import transforms
from metrics import compute_metric, success_rate
normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
transform = transforms.Compose([
        # transforms.Resize((224,224)),
        #transforms.ToTensor(),
        transforms.Normalize(*normalization)
    ])

'''
we update each time, no condition
'''
def multi(model, dict, hyper, logger):
    print('enter multi')
    eps, lr, bs, beta, iterations, norm_threshold, d, e = hyper['eps'], hyper['lr'], hyper['bs'], hyper['beta'],\
                                                       hyper['iters'], hyper['norm_threshold'], hyper['direction'], hyper['weights']
    ori_steer_pred, ori_throttle_pred = torch.tensor(dict['steer']), torch.tensor(dict['speed'])
    ori_imgs, ori_opticals, raw_imgs = dict['ori_imgs'], dict['ori_opticals'], dict['raw_imgs']
    total_perturb = torch.zeros((3, 224, 224))
    imgs_ori = copy.deepcopy(ori_imgs)
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(ori_imgs)
    real_total_perturb_iterlist, adversarial = [], []
   
    for i in range(iterations):
        if i%50==0 and i != 0:
            lr = lr*0.8
        torch.cuda.empty_cache()
        np.random.shuffle(idx)
        for j in range(0,len(imgs),bs):
            torch.cuda.empty_cache()
            num=0
            minibatch={}
            if j <= len(imgs) - bs:
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, j + bs)]
                minibatch['opticals'] = [ori_opticals[idx[k]] for k in range(j, j + bs)]
                perturb = torch.zeros((bs, 3, 224, 224))

            else:  # create mini-batch
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, len(imgs))]
                minibatch['opticals'] = [ori_opticals[idx[k]] for k in range(j, len(imgs))]
                perturb = torch.zeros((len(imgs) - j, 3, 224, 224))
       
            #for each sample
            for image, optical in zip(minibatch['rgb'], minibatch['opticals']):
                image.requires_grad = True
                angle_hat, speed_hat = model(image.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                             optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
                steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]

                #now minimize the loss
                lossA, lossB = 1 / beta * torch.exp(steer_pred * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                    speed_pred * (-1 / beta) * d[1])
                loss = e[0]*lossA+e[1]*lossB
                loss.backward()
                perturb[num] = image.grad.data
                num += 1
                image.grad.data.zero_()
                

            if hyper['strategy'] == 'mean':
                combine = torch.mean(perturb, dim=0)

            if hyper['strategy'] == 'max':
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]

            if torch.norm(combine) < norm_threshold and torch.norm(combine)!=0:
                combine = combine * norm_threshold / torch.norm(combine)
            
            
            total_perturb = torch.clamp(total_perturb - lr*combine, min=-eps/255, max=eps/255)
            total_perturb255 = (total_perturb * 255)
            adv_imgs = torch.clamp(torch.add(raw_imgs, total_perturb255[None, ...]), min=0,
                                   max=255)
            adv_imgs = transform(adv_imgs / 255)
            with torch.no_grad():
                adv_steer, adv_speed = [], []
                for img, optical in zip(adv_imgs, ori_opticals):
                    angle_hat, speed_hat = model(img.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                                 optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
                    steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
                    adv_steer.append(steer_pred.data.cpu().numpy())
                    adv_speed.append(speed_pred.data.cpu().numpy())
                adv_steer_vec, adv_throttle_vec = np.array(adv_steer), np.array(adv_speed)

            imgs = copy.deepcopy(adv_imgs.cpu())
            real_total_perturb_iterlist.append(total_perturb)

            adversarial.append([adv_steer_vec, adv_throttle_vec])  # in radius

        s_adv, t_adv = adversarial[-1][0], adversarial[-1][1]
        serror, terror = (s_adv - ori_steer_pred.numpy()) * d[0], (t_adv - ori_throttle_pred.numpy()) * d[1]
        compute_metric(hyper, serror, terror, logger)
        success_rate(serror, terror, logger)
    return real_total_perturb_iterlist, adversarial


def multi_gradnorm(model, dict, hyper, logger):
    eps, lr, bs, beta, alpha, lrgrad, iterations, norm_threshold, d = hyper['eps'], hyper['lr'], hyper['bs'], hyper[
        'beta'], hyper['alpha'], hyper['lrgrad'], hyper['iters'], hyper['norm_threshold'], hyper['direction']
    scale = hyper['scale']
    ori_steer_pred, ori_throttle_pred = torch.tensor(dict['steer']), torch.tensor(dict['speed'])
    ori_imgs, ori_opticals, raw_imgs = dict['ori_imgs'], dict['ori_opticals'], dict['raw_imgs']
    init_loss_s = torch.mean(1 / beta * torch.exp(ori_steer_pred * (-1 / beta) * d[0]))
    init_loss_t = torch.mean(1 / beta * torch.exp(ori_throttle_pred/scale * (-1 / beta) * d[1]))

    initial_task_loss = torch.tensor([init_loss_s, init_loss_t])
    print('initial task loss', initial_task_loss)  # tensor([5.3258e-01, 8.0294e-05]
    total_perturb = torch.zeros((3, 224, 224))
    imgs_ori = copy.deepcopy(ori_imgs)  # all original images in this video
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(ori_imgs)   #[22,3,224,224] preprocessed img, not in (0,1)
    real_total_perturb_iterlist, adversarial = [], []
    a1, a2 = Variable(torch.tensor(1/2), requires_grad=True).cuda(), Variable(torch.tensor(1/2), requires_grad=True).cuda()
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
                minibatch['opticals'] = [ori_opticals[idx[k]] for k in range(j, j + bs)]
                perturb = torch.zeros((bs, 3, 224, 224))

            else:  # create mini-batch
                minibatch['rgb'] = [imgs[idx[k]] for k in range(j, len(imgs))]
                minibatch['opticals'] = [ori_opticals[idx[k]] for k in range(j, len(imgs))]
                perturb = torch.zeros((len(imgs) - j, 3, 224, 224))
            total_norms = torch.tensor([[0.0, 0.0]]).cuda()
            unweighted_losses = []

            # for each data sample
            for image, optical in zip(minibatch['rgb'], minibatch['opticals']):  # for a single image in mini-batch
                image.requires_grad = True
                angle_hat, speed_hat = model(image.repeat(5,1,1,1).unsqueeze(0).cuda(), optical.repeat(5,1,1,1).unsqueeze(0).cuda()) #[1,5,3,224,224]
                steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
                lossA, lossB = 1 / beta * torch.exp(steer_pred * (-1 / beta) * d[0]), 1 / beta * torch.exp(
                    speed_pred/scale * (-1 / beta) * d[1])

                loss = a1 * lossA + a2 * lossB
                loss.backward(retain_graph=True)
                unweighted_losses.append([lossA.data.cpu(), lossB.data.cpu()])

                perturb[num] = image.grad.data
                num = num + 1

                single_norm = torch.tensor([0.0]).cuda()

                for t, a in zip([lossA, lossB], [a1, a2]):
                    gygw = torch.autograd.grad(t, image, retain_graph=True)
                    cop = torch.unsqueeze(torch.norm(torch.mul(a, gygw[0].cuda())), dim=0)

                    single_norm = torch.cat((single_norm, cop), 0)

                single_norm = single_norm[1:]

                total_norms = torch.cat((total_norms, torch.unsqueeze(single_norm, dim=0)), 0)
                image.grad.data.zero_()

            total_norms = total_norms[1:]
            norms = torch.mean(total_norms, dim=0)
            task_loss = torch.mean(torch.tensor(unweighted_losses), dim=0)
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss

            inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
            mean_norm = np.mean(norms.data.cpu().numpy())
            constant_term = (mean_norm * (inverse_train_rate ** alpha)).clone().detach().cuda()
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
            a1.grad = torch.autograd.grad(grad_norm_loss, a1, retain_graph=True)[0]
            a2.grad = torch.autograd.grad(grad_norm_loss, a2, retain_graph=True)[0]

            tmp_a1, tmp_a2 = a1 - lrgrad * a1.grad, a2 - lrgrad * a2.grad
            if tmp_a1 > 0:  # update only if new a_i is positive
                a1 = tmp_a1
            if tmp_a2 > 0:
                a2 = tmp_a2
            # renormalize sum to 1
            normalize_coeff = 1 / (a1 + a2)
            a1, a2 = a1 * normalize_coeff, a2 * normalize_coeff
            if hyper['strategy'] == 'mean':
                combine = torch.mean(perturb, dim=0)

            if hyper['strategy'] == 'max':
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]

            if torch.norm(combine) < norm_threshold and torch.norm(combine) != 0:
                combine = combine * norm_threshold / torch.norm(combine)

            total_perturb = torch.clamp(total_perturb - lr * combine, min=-eps/255, max=eps/255)
            total_perturb255 = (total_perturb * 255) #0~255
             #0~255
            adv_imgs = torch.clamp(torch.add(raw_imgs, total_perturb255[None, ...]), min=0,
                                   max=255)
            adv_imgs = transform(adv_imgs/255)
            with torch.no_grad():
                adv_steer, adv_speed = [], []
                for img, optical in zip(adv_imgs, ori_opticals):
                    angle_hat, speed_hat = model(img.repeat(5,1,1,1).unsqueeze(0).cuda(), optical.repeat(5,1,1,1).unsqueeze(0).cuda())
                    steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
                    adv_steer.append(steer_pred.data.cpu().numpy())
                    adv_speed.append(speed_pred.data.cpu().numpy())
                adv_steer_vec, adv_throttle_vec = np.array(adv_steer), np.array(adv_speed)

            # update all images with no condition

            imgs = copy.deepcopy(adv_imgs.cpu())
            real_total_perturb_iterlist.append(total_perturb)
            adversarial.append([adv_steer_vec, adv_throttle_vec])

        s_adv, t_adv = adversarial[-1][0], adversarial[-1][1]
        serror, terror = (s_adv - ori_steer_pred.numpy()) * d[0], (t_adv - ori_throttle_pred.numpy()) * d[1]

        compute_metric(hyper, serror, terror, logger)
        success_rate(serror, terror, logger)

    return real_total_perturb_iterlist, adversarial







