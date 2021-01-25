"""
Causal metrics (insertion/deletion game) from https://github.com/eclique/RISE/blob/master/evaluation.py
"""

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from torchray.benchmark.datasets import COCO_CLASSES as classes

"""
Mostly adapted from https://github.com/eclique/RISE/blob/master/evaluation.py"""
"""
Mostly adapted from https://github.com/eclique/RISE/blob/master/evaluation.py"""


def tensor_imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def auc(arr):
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def blur(x, klen=11, ksig=5):
    kern = gkern(klen, ksig)
    return nn.functional.conv2d(x, kern, padding=klen // 2)


def causal_metric(model,
                  sal,
                  input,
                  target,
                  verbose=0,
                  mode='del',
                  steps=None, save_as=''):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev)

    with torch.no_grad():

        HW = sal.shape[-1] * sal.shape[-2]

        if steps is None:
            steps = 96

        step = HW // steps
        num_steps = (HW + step - 1) // step

        _, c, h, w = input.shape

        masked_imgs = np.zeros((num_steps, c, h, w))

        if mode == 'del':
            title = 'Deletion game'
            xlabel = 'Pixels deleted (in order of saliency)'
            start = input.clone().reshape(1, 3, HW).cpu().numpy()
            finish = torch.zeros_like(input).reshape(1, 3, HW).cpu().numpy()
        else:
            title = 'Insertion game'
            xlabel = 'Pixels inserted (in order of saliency)'
            start = blur(input.cpu()).to(dev).reshape(1, 3, HW).cpu().numpy()
            finish = input.clone().reshape(1, 3, HW).cpu().numpy()

        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(sal.cpu().numpy().reshape(-1, HW), axis=1), axis=-1)

        for i in range(num_steps):
            coords = salient_order[:, step * i:step * (i + 1)]
            masked_img = start
            masked_img[0, :, coords] = finish[0, :, coords]
            masked_imgs[i] = masked_img.reshape(1, c, h, w)

        imgs_tensor = torch.Tensor(masked_imgs).to(dev)
        scores = normalise(model(imgs_tensor)[:, target].reshape(-1))

        if verbose > 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(np.arange(num_steps) / num_steps, scores.cpu())
            plt.xlim(-0.1, 1.1)
            plt.ylim(0, 1.05)
            plt.fill_between(np.arange(num_steps) / num_steps, 0, scores.cpu(), alpha=0.4)
            plt.title(title)
            plt.ylabel("Target class confidence")
            plt.xlabel(xlabel)
            plt.subplot(122)
            plt.imshow(normalise(input[0].cpu().numpy().transpose(1, 2, 0)))
            plt.axis('off')

            plt.savefig('data/attribution_causal_benchmarks/pngs/{}_{}_{}-n.png'.format(save_as, classes[target], mode))
            # plt.show()
            plt.clf()

    return scores



def normalise(x):
    x = x + abs(x.min())
    min_x = x.min()
    max_x = x.max()
    range_x = max_x - min_x
    if range_x == 0:
        return x
    return (x - min_x) / range_x