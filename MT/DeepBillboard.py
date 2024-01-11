'''
DeepBillboard
'''
import torch
import numpy as np
import copy
import random
import math
from torchvision import transforms
normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Normalize(*normalization)
    ])

def find_kth_max(array, k):
    tmp = array.flatten()
    tmp = abs(tmp)
    tmp.sort()
    return tmp[-k]


def deepbillboard(model, dict, hyper):
    strategy, eps, lr, bs, iterations, ds = hyper['strategy'], hyper['eps'], hyper['lr'], hyper['bs'], hyper['iters'], hyper['direction'][0]

    ori_steer_pred, ori_throttle_pred = torch.tensor(dict['steer']), torch.tensor(dict['speed'])
    ori_imgs, ori_opticals, raw_imgs = dict['ori_imgs'], dict['ori_opticals'], dict['raw_imgs']
    total_perturb = torch.zeros((3, 224, 224))
    imgs_ori = copy.deepcopy(ori_imgs)
    idx = np.arange(len(imgs_ori))
    imgs = copy.deepcopy(ori_imgs)
    real_total_perturb_iterlist, adversarial = [], []
    last_diff = 0

    for i in range(iterations):
        if i % 50 == 0 and i != 0:
            lr = lr * 0.8
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

            # for each sample
            for image, optical in zip(minibatch['rgb'], minibatch['opticals']):
                image.requires_grad = True
                angle_hat, speed_hat = model(image.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                             optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
                steer_pred, _ = angle_hat.flatten()[-1], speed_hat.flatten()[-1]

                loss = -steer_pred * ds
                loss.backward()
                grads = image.grad.data.clone()
                if hyper['jsma']:
                    k_th_value = find_kth_max(grads, hyper['jsma_n'])
                    super_threshold_indices = abs(grads) < k_th_value
                    grads[super_threshold_indices] = 0
                perturb[num] = grads
                num += 1
                image.grad.data.zero_()

            # now combine batch-size perturbation proposals into a single
            if strategy == 'max':
                indexa = torch.max(torch.abs(perturb), dim=0, keepdim=True)[1]
                combine = torch.gather(perturb, dim=0, index=indexa)[0]
            if strategy == 'sum':
                combine = torch.sum(perturb, dim=0)

            tmp_total_perturb = torch.clamp(total_perturb - lr * combine, min=-eps/255, max=eps/255)
            total_perturb255 = (total_perturb * 255)

            tmp_adv_imgs = torch.clamp(torch.add(raw_imgs, total_perturb255[None, ...]), min=0,
                                   max=255)
            tmp_adv_imgs = transform(tmp_adv_imgs / 255)
            with torch.no_grad():
                adv_steer, adv_speed = [], []
                for img, optical in zip(tmp_adv_imgs, ori_opticals):
                    angle_hat, speed_hat = model(img.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                                 optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
                    steer_pred, speed_pred = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
                    adv_steer.append(steer_pred.data.cpu().numpy())
                    adv_speed.append(speed_pred.data.cpu().numpy())
                adv_steer_vec, adv_throttle_vec = np.array(adv_steer), np.array(adv_speed)
            tmp_error_s = np.sum(np.abs(adv_steer_vec - ori_steer_pred.numpy()))

            this_diff = tmp_error_s
            if this_diff > last_diff:
                total_perturb = copy.deepcopy(tmp_total_perturb)
                last_diff = this_diff
                imgs = copy.deepcopy(tmp_adv_imgs)
                real_total_perturb_iterlist.append(total_perturb)
                adversarial.append([adv_steer_vec, adv_throttle_vec])
            else:
                if hyper['simulated_annealing']:
                    if (random.random() < pow(math.e, hyper['sa_k'] * (this_diff - last_diff) / (
                    pow(hyper['sa_b'], i))) and this_diff != last_diff):
                        total_perturb = copy.deepcopy(tmp_total_perturb)
                        imgs = copy.deepcopy(tmp_adv_imgs)
                        last_diff = this_diff
                        real_total_perturb_iterlist.append(total_perturb)
                        adversarial.append([adv_steer_vec, adv_throttle_vec])
    return real_total_perturb_iterlist, adversarial