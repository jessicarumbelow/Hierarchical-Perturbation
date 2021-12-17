import torch
import numpy as np
from torchray.utils import imsc
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import pandas as pd


def gkern(klen, nsig):
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def blur(x, klen=11, ksig=5):
    kern = gkern(klen, ksig)
    return F.conv2d(x, kern, padding=klen // 2)


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def hierarchical_perturbation(model,
                              input,
                              target,
                              vis=False,
                              interp_mode='nearest',
                              resize=None,
                              batch_size=32,
                              perturbation_type='mean', threshold_mode='mid-range', return_info=False, diff_func=torch.relu, depth_bound=2, verbose=True):
    if verbose: print('\nBelieve the HiPe!')
    with torch.no_grad():

        dev = input.device
        if dev == 'cpu':
            batch_size=1
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        num_cells = int(max(np.ceil(np.log2(dim)), 1)/2)
        max_depth = int(np.log2(dim / num_cells)) - depth_bound
        if verbose: print('Max depth: {}'.format(max_depth))
        saliency = torch.zeros((1, 1, input_y_dim, input_x_dim), device=dev)
        max_batch = batch_size

        thresholds_d_list = []
        masks_d_list = []

        output = model(input)[:, target]

        if perturbation_type == 'blur':
            pre_b_image = blur(input.clone().cpu()).to(dev)

        while depth <= max_depth:

            masks_list = []
            b_list = []
            num_cells *= 2
            depth += 1
            if threshold_mode == 'mean':
                threshold = torch.mean(saliency)
            else:
                threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            thresholds_d_list.append(diff_func(threshold).item())

            y_ixs = range(-1, num_cells)
            x_ixs = range(-1, num_cells)
            x_cell_dim = input_x_dim // num_cells
            y_cell_dim = input_y_dim // num_cells

            
            if verbose: 
                print('Depth: {}, {} x {} Cell Dim'.format(depth, y_cell_dim, x_cell_dim))
                print('Threshold: {}'.format(threshold))
                print('Range {:.1f} to {:.1f}'.format(saliency.min(), saliency.max()))
            possible_masks = 0

            for x in x_ixs:
                for y in y_ixs:
                    possible_masks += 1
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(x + 2, num_cells), min(y + 2, num_cells)

                    mask = torch.zeros((1, 1, num_cells, num_cells), device=dev)
                    mask[:, :, y1:y2, x1:x2] = 1.0
                    local_saliency = F.interpolate(mask, (input_y_dim, input_x_dim), mode=interp_mode) * saliency

                    if depth > 1:
                        local_saliency = torch.max(diff_func(local_saliency))
                    else:
                        local_saliency = 0

                    # If salience of region is greater than the average, generate higher resolution mask
                    if local_saliency >= threshold:

                        masks_list.append(abs(mask - 1))

                        if perturbation_type == 'blur':

                            b_image = input.clone()
                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim]
                            b_list.add(b_image)

                        if perturbation_type == 'mean':
                            b_image = input.clone()
                            mean = torch.mean(b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
                                              axis=(-1, -2), keepdims=True)

                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = mean
                            b_list.append(b_image)

            num_masks = len(masks_list)
            if verbose: print('Selected {}/{} masks at depth {}'.format(num_masks, possible_masks, depth))
            if num_masks == 0:
                depth -= 1
                break
            total_masks += num_masks
            masks_d_list.append(num_masks)

            while len(masks_list) > 0:
                m_ix = min(len(masks_list), max_batch)
                if perturbation_type != 'fade':
                    b_imgs = torch.cat(b_list[:m_ix])
                    del b_list[:m_ix]
                masks = torch.cat(masks_list[:m_ix])
                del masks_list[:m_ix]

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':
                    perturbed_inputs = input * masks
                    perturbed_outputs = diff_func(output - model(input * masks)[:, target])
                else:
                    perturbed_outputs = diff_func(output - model(b_imgs)[:, target])
    
                if len(list(perturbed_outputs.shape)) == 1:
                    sal = perturbed_outputs.reshape(-1,1,1,1) * torch.abs(masks - 1)
                else:
                    sal = perturbed_outputs * torch.abs(masks - 1)
                    
                saliency += torch.sum(sal, dim=(0, 1))

            if vis:
                plt.figure(figsize=(8, 4))
                plt.title('Depth: {}, {} x {} Mask\nThreshold: {:.1f}'.format(depth, num_cells, num_cells, threshold))
                plt.imshow(np.transpose((normalise(input) * normalise(saliency))[0], (1, 2, 0)))
                plt.show()
                #plt.figure(figsize=(8, 4))
                #pd.Series(normalise(saliency).reshape(-1)).plot(label='Saliency ({})'.format(threshold_mode))
                #pd.Series(normalise(input).reshape(-1)).plot(label='Actual')
                #plt.legend()
                #plt.show()

        if verbose: print('Used {} masks in total.'.format(total_masks))
        saliency = normalise(saliency)
        if resize is not None:
            saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
        if return_info:
            return saliency, {'thresholds': thresholds_d_list, 'masks': masks_d_list, 'total_masks': total_masks}
        else:
            return saliency, total_masks
