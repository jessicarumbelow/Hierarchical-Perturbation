import torch
import numpy as np
from torchray.utils import imsc
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import time
import inspect

def perturbation_search_alternate(model,
                                  original_input,
                                  c,
                                  vis=False,
                                  interp_mode='nearest',
                                  resize=None,
                                  perturbation_type='fade'):
    with torch.no_grad():

        dev = original_input.device
        unoccluded_output = model(original_input)[0][c]

        _, channels, input_y_dim, input_x_dim = original_input.shape
        y_ix, x_ix = 0, 0
        y_dim = 2 ** int(np.ceil(np.log2(input_y_dim)))
        x_dim = 2 ** int(np.ceil(np.log2(input_x_dim)))
        x_dim, y_dim = max(x_dim, y_dim), max(x_dim, y_dim)

        input = torch.zeros((_, channels, y_dim, x_dim), device=dev)
        input[:, :, :input_y_dim, :input_x_dim] = original_input
        num_occs = 0
        depth = 0
        branches = [(x_dim, x_ix, y_dim, y_ix, depth)]
        max_depth = int(np.log2(y_dim))
        min_k = max_depth
        saliency_maps = torch.zeros((y_dim, x_dim), device=dev)
        blank_mask = torch.ones((_, channels, y_dim, x_dim), device=dev)
        blurred_input = blur(input.clone().cpu()).to(dev)

        while len(branches) > 0:
            x_dim, x_ix, y_dim, y_ix, depth = branches.pop(0)

            if x_ix < input_x_dim and y_ix < input_y_dim:
                occluded_input = input.clone()

                if perturbation_type == 'fade':
                    mask = blank_mask.clone()
                    mask[:, :, y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = 0.0
                    occluded_input = occluded_input * mask

                elif perturbation_type == 'blur':
                    occluded_input[:, :, y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = blurred_input[:, :, y_ix:y_ix + y_dim, x_ix:x_ix + x_dim]
                else:
                    m = torch.mean(occluded_input[:, :, y_ix:y_ix + y_dim - max(0, y_ix + y_dim - input_y_dim), x_ix:x_ix + x_dim - max(0, x_ix + x_dim - input_x_dim)][0], axis=(-1, -2),
                                   keepdims=True)
                    occluded_input[:, :, y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = m

                occluded_output = model(occluded_input[:, :, :input_y_dim, :input_x_dim])[0][c]

                output_difference = unoccluded_output - occluded_output
                num_occs += 1

                threshold = (unoccluded_output / (4 ** depth)) if (depth > 0) else output_difference

                # print('Occlusions: {}\nDepth: {}\nThreshold: {}\nOutput Difference: {}'.format(num_occs, depth, threshold, output_difference))

                if output_difference >= threshold:

                    saliency_maps[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] += output_difference * (4 ** depth)

                    if (x_dim / 2 >= min_k) and (y_dim / 2 >= min_k):
                        x_dim = x_dim // 2
                        y_dim = y_dim // 2
                        branches.extend([
                            (x_dim, x_ix, y_dim, y_ix, depth + 1),
                            (x_dim, x_ix + x_dim, y_dim, y_ix, depth + 1),
                            (x_dim, x_ix, y_dim, y_ix + y_dim, depth + 1),
                            (x_dim, x_ix + x_dim, y_dim, y_ix + y_dim, depth + 1),
                            ])

        if resize is not None:
            saliency = F.interpolate(saliency_maps[:input_y_dim, :input_x_dim].unsqueeze(0).unsqueeze(0), (resize[1], resize[0]), mode=interp_mode)

        return saliency, num_occs


class Timer:

    def __init__(self):
        self.last_time = time.process_time()
        self.last_line = 0
        self.last_note = ''

    def timer(self, note=''):
        cur_time = time.process_time()
        ln = inspect.currentframe().f_back.f_lineno
        diff = f"{cur_time - self.last_time:.10}"

        print(self.last_line, ' to ', ln, ' took ', diff)
        print('({} to {})\n'.format(self.last_note, note))
        self.last_note = note
        self.last_line = ln
        self.last_time = cur_time


def perturbation_search(model,
                        input,
                        target,
                        vis=False,
                        interp_mode='nearest',
                        resize=None,
                        batch_size=32,
                        perturbation_type='fade'):

    with torch.no_grad():

        # Get device of input (i.e., GPU).
        dev = input.device
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        num_cells = 4
        max_depth = int(np.log2(dim / num_cells)) - 2
        saliency = torch.zeros((1, 1, input_y_dim, input_x_dim), device=dev)
        max_batch = batch_size

        output = model(input)[:, target]

        if perturbation_type == 'blur':
            pre_b_image = blur(input.clone().cpu()).to(dev)

        while depth < max_depth:

            masks_list = []
            b_list = []
            num_cells *= 2
            depth += 1
            threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            #print('Depth: {}, {} x {} Cell'.format(depth, input_y_dim//num_cells, input_x_dim//num_cells))
            # print('Threshold: {:.1f}'.format(threshold))
            # print('Range {:.1f} to {:.1f}'.format(saliency.min(), saliency.max()))

            y_ixs = range(-1, num_cells)
            x_ixs = range(-1, num_cells)
            x_cell_dim = input_x_dim // num_cells
            y_cell_dim = input_y_dim // num_cells

            pos_masks = 0

            for x in x_ixs:
                for y in y_ixs:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(x + 2, num_cells), min(y + 2, num_cells)
                    pos_masks += 1

                    mask = torch.zeros((1, 1, num_cells, num_cells), device=dev)
                    mask[:, :, y1:y2, x1:x2] = 1.0
                    local_saliency = F.interpolate(mask, (input_y_dim, input_x_dim), mode=interp_mode) * saliency

                    if depth > 1:
                        local_saliency = torch.max(local_saliency)
                    else:
                        local_saliency = 0

                    # If salience of region is greater than the average, generate higher resolution mask
                    if local_saliency >= threshold:

                        masks_list.append(abs(mask - 1))

                        if perturbation_type == 'blur':

                            b_image = input.clone()
                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim]
                            b_list.append(b_image)

                        if perturbation_type == 'mean':
                            b_image = input.clone()
                            mean = torch.mean(b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
                                           axis=(-1, -2), keepdims=True)

                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = mean
                            b_list.append(b_image)

            num_masks = len(masks_list)
            #print('Selected {}/{} masks at depth {}'.format(num_masks, pos_masks, depth))

            if num_masks == 0:
                depth -= 1
                break
            total_masks += num_masks

            while len(masks_list) > 0:
                # print('Processing {} masks'.format(len(masks_list)))
                m_ix = min(len(masks_list), max_batch)
                if perturbation_type != 'fade':
                    b_imgs = torch.cat(b_list[:m_ix])
                    del b_list[:m_ix]
                masks = torch.cat(masks_list[:m_ix])
                del masks_list[:m_ix]

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':
                    perturbed_outputs = torch.relu(output - model(input * masks)[:, target])
                else:
                    perturbed_outputs = torch.relu(output - model(b_imgs)[:, target])

                sal = perturbed_outputs * torch.abs(masks.transpose(0, 1) - 1)
                saliency += torch.sum(sal, dim=(0, 1))

                if vis:
                    print('Saving image...')
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.title('Depth: {}, {} x {} Mask\nThreshold: {:.1f}'.format(depth, num_cells, num_cells, threshold))
                    # imsc((input * masks)[0])
                    imsc(b_imgs[0])
                    plt.subplot(1, 2, 2)
                    imsc(saliency[0])
                    plt.show()
                    plt.savefig('data/attribution_benchmarks/Preview')
                    plt.close()

        # print('Used {} masks in total.'.format(total_masks))
        if resize is not None:
            saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
        return saliency, total_masks


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
    return F.conv2d(x, kern, padding=klen // 2)
