import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Normalize(*normalization)
    ])

def fgsmattack(model, ori_img, raw_img, optical, hyper, w, logger):
    iterations, beta, lr, d = hyper['iters'], hyper['beta'], hyper['lr'], hyper['direction']
    w = torch.tensor(w).type(torch.FloatTensor).cuda()
    steer_iterlist, throttle_iterlist = [], []

    real_total_perturb_iterlist = []
    image = Variable(ori_img, requires_grad=True)

    for i in range(iterations):
        angle_hat, speed_hat = model(image.repeat(5, 1, 1, 1).unsqueeze(0).cuda(),
                                     optical.repeat(5, 1, 1, 1).unsqueeze(0).cuda())
        adv_steer, adv_throttle = angle_hat.flatten()[-1], speed_hat.flatten()[-1]
        steer_iterlist.append(adv_steer.data.cpu().numpy())
        throttle_iterlist.append(adv_throttle.data.cpu().numpy())
        lossA, lossB = 1 / beta * torch.exp(adv_steer * (-1 / beta) * d[0]), 1 / beta * torch.exp(
            adv_throttle * (-1 / beta) * d[1])
        loss = w[0] * lossA + w[1] * lossB
        loss.backward()

        perturbation = lr * torch.sign(image.grad.data)
        perturb255 = perturbation * 255

        image.data = torch.clamp(raw_img - perturb255, min=0, max=255)
        image.data = transform(image.data/255)
        image.grad.data.zero_()

    return steer_iterlist, throttle_iterlist, real_total_perturb_iterlist



def run_fgsm(hyper, model, dict, logger):
    d = hyper['direction']
    ori_imgs, ori_opticals, raw_imgs = dict['ori_imgs'], dict['ori_opticals'], dict['raw_imgs']

    serror_framelist, terror_framelist = [], []
    if hyper['weight_strategy'] == 'equal':
        w = [1/2, 1/2]
    for i, (image, optical, raw_img) in enumerate(zip(ori_imgs, ori_opticals, raw_imgs)):  # load a single image frame
        s_iterlist, t_iterlist, perturb_iter = fgsmattack(model, image, raw_img, optical, hyper, w, logger)
        t_error = (t_iterlist[-1] - t_iterlist[0]) * d[1]
        s_error = (s_iterlist[-1] - s_iterlist[0]) * d[0]
        serror_framelist.append(s_error)
        terror_framelist.append(t_error)
    return serror_framelist, terror_framelist
