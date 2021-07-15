import torch
import numpy as np
from torchray.utils import imsc
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import pandas as pd


num_el = 256

f_sizes = [4,8,16,32,64]

def model(x):
    x = torch.sum(x, dim=(-1,-2), keepdim=True)
    return x

for f in f_sizes:
    input = gkern(num_el, f)[0,0].reshape(1,1,num_el, num_el)

    sal_rise = rise(model, input)
    sal_rise = sal_rise[:, 0, :, :].unsqueeze(1)

    sal_midrange, info_midrange = hierarchical_perturbation(model, input, 0, vis=False, perturbation_type='fade', batch_size=32, interp_mode='nearest', return_info=True)
    sal_mean, info_mean = hierarchical_perturbation(model, input, 0, vis=False, perturbation_type='fade', batch_size=32, interp_mode='nearest', threshold_mode='mean', return_info=True)

    input2 = input[:,:,num_el//2]
    sal_rise2 = sal_rise[:,:,num_el//2]
    sal_midrange2 = sal_midrange[:,:,num_el//2]
    sal_mean2 = sal_mean[:,:,num_el//2]

    t_mean, ms_mean, m_mean = info_mean.values()
    t_midrange, ms_midrange, m_midrange = info_midrange.values()
    print('Mean masks: {}, Mid-range masks: {}'.format(m_mean, m_midrange))
    print('\nComparison of HiPe with mean threshold, HiPe with mid-range threshold, and RISE\nSalient feature diameter {}'.format(f))
    plt.figure(figsize=(17, 8))
    ax = plt.subplot(1, 4, 1)
    ax.set_title("Actual Saliency")
    imsc(input[0])
    ax = plt.subplot(1, 4, 2)
    ax.set_title("Estimated Saliency (HiPe, mid-range)")
    imsc(sal_midrange[0])
    ax = plt.subplot(1, 4, 3)
    ax.set_title("Estimated Saliency (HiPe, mean)")
    imsc(sal_mean[0])
    ax = plt.subplot(1, 4, 4)
    ax.set_title("Estimated Saliency (RISE)")
    imsc(sal_rise[0])
    plt.show()


    plt.figure(figsize=(16, 4))
    pd.Series(normalise(sal_mean2).reshape(-1)).plot(label='Saliency ({})'.format('HiPe, mean threshold'))
    pd.Series(normalise(sal_midrange2).reshape(-1)).plot(label='Saliency ({})'.format('HiPe, mid-range threshold'))
    pd.Series(normalise(sal_rise2).reshape(-1)).plot(label='Saliency ({})'.format('RISE'))
    pd.Series(normalise(input2).reshape(-1)).plot(label='Actual')
    plt.xlabel('Element position')
    plt.ylabel('Saliency')
    plt.title('Flattened Saliency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 2, 1)
    pd.Series(t_mean).plot(label='Thresholds ({})'.format('HiPe, mean threshold'))
    pd.Series(t_midrange).plot(label='Thresholds ({})'.format('HiPe, mid-range threshold'))
    plt.xlabel('Depth')
    plt.ylabel('Saliency Threshold')
    plt.title('Saliency Threshold Comparison')
    plt.legend()
    ax = plt.subplot(1,2,2)
    pd.Series(ms_mean).plot(label='Masks ({})'.format('HiPe, mean threshold'))
    pd.Series(ms_midrange).plot(label='Masks ({})'.format('HiPe, mid-range threshold'))
    plt.xlabel('Depth')
    plt.ylabel('Number of masks')
    plt.title('Number of Masks Comparison')
    plt.legend()
    plt.show()