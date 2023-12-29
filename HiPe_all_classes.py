import torch
import numpy as np
from torchray.utils import imsc
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import pandas as pd


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


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def hierarchical_perturbation(model, input, interp_mode='nearest', resize=None, perturbation_type='mean',
							  threshold_mode='mid-range', return_info=False, diff_func=torch.relu, max_depth=2, verbose=True,
							  cell_init=2):
	if verbose: print('\nBelieve the HiPe!')
	with torch.no_grad():
		dev = input.device
		if verbose: print('Using device: {}'.format(dev))
		bn, channels, input_y_dim, input_x_dim = input.shape
		dim = min(input_x_dim, input_y_dim)
		total_masks = 0
		depth = 0
		num_cells = int(max(np.ceil(np.log2(dim)), 1) / cell_init)
		base_max_depth = int(np.log2(dim / num_cells)) - 2
		if max_depth == -1 or max_depth > base_max_depth + 2:
			max_depth = base_max_depth
		if verbose: print('Max depth: {}'.format(max_depth))
    
		def identity(x):
			return x

		if diff_func == None:
			diff_func = identity

		thresholds_d_list = []
		masks_d_list = []

		output = model(input)[0]

		num_classes = output.shape[0]
		saliency = torch.zeros((1, num_classes, input_y_dim, input_x_dim), device=dev)

		if perturbation_type == 'blur':
			pre_b_image = blur(input.clone().cpu()).to(dev)

		while depth < max_depth:
			masks_list = []
			b_list = []
			num_cells *= 2
			depth += 1
			if threshold_mode == 'var':
				threshold = torch.amin(saliency, dim=(-1, -2)) + (
						(torch.amax(saliency, dim=(-1, -2)) - torch.amin(saliency, dim=(-1, -2))) / 2)
				threshold = -torch.var(threshold)
			elif threshold_mode == 'mean':
				threshold = torch.mean(saliency)
			else:
				threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

			if verbose:
				print('Threshold: {}'.format(threshold))
			thresholds_d_list.append(diff_func(threshold))

			y_ixs = range(-1, num_cells)
			x_ixs = range(-1, num_cells)
			x_cell_dim = input_x_dim // num_cells
			y_cell_dim = input_y_dim // num_cells

			if verbose:
				print('Depth: {}, {} x {} Cell Dim'.format(depth, y_cell_dim, x_cell_dim))
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
						if threshold_mode == 'var':
							local_saliency = -torch.var(torch.amax(local_saliency, dim=(-1, -2)))
						else:
							local_saliency = torch.max(diff_func(local_saliency))
					else:
						local_saliency = 0

					# If salience of region is greater than the average, generate higher resolution mask
					if local_saliency >= threshold:
						masks_list.append(abs(mask - 1))

						if perturbation_type == 'blur':
							b_image = input.clone()
							b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
							x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
															   x1 * x_cell_dim:x2 * x_cell_dim]
							b_list.append(b_image)

						if perturbation_type == 'mean':
							b_image = input.clone()
							mean = torch.mean(
								b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
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
				if perturbation_type != 'fade':
					b_imgs = b_list.pop()
				masks = masks_list.pop()

				# resize low-res masks to input size
				masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

				if perturbation_type == 'fade':
					perturbed_outputs = diff_func(output - model(input * masks)[0][0])
				else:
					perturbed_outputs = diff_func(output - model(b_imgs)[0][0])

				if len(list(perturbed_outputs.shape)) == 1:
					sal = perturbed_outputs.reshape(-1, 1, 1, 1) * torch.abs(masks - 1)
				else:
					sal = perturbed_outputs.reshape(1, num_classes, 1, 1) * torch.abs(masks - 1)

				saliency += sal

		if verbose: print('Used {} masks in total.'.format(total_masks))
		if resize is not None:
			saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
		if return_info:
			return saliency, {'thresholds': thresholds_d_list, 'masks': masks_d_list, 'total_masks': total_masks}
		else:
			return saliency, total_masks

