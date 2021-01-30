import os

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset

from torchray.attribution.common import get_pointing_gradient
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.excitation_backprop import update_resnet
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.rise import rise
from torchray.benchmark.datasets import get_dataset, coco_as_mask, voc_as_mask
from torchray.benchmark.models import get_model, get_transform
from torchray.utils import imsc, get_device, xmkdir
import torchray.attribution.extremal_perturbation as elp
from HiPe import hierarchical_perturbation, hierarchical_perturbation_alternate
from torchray.benchmark.datasets import COCO_CLASSES as classes
import time
import torch.nn.functional as F
from causal_metrics import *

series = 'attribution_causal_benchmarks'
series_dir = os.path.join('data', series)
seed = 0
chunk = None
vis = False
lim = 1000

datasets = ['coco', 'voc_2007']

"""['voc_2007',
    'coco'
    ]
"""

archs = ['resnet50']

hipe_experiment = 'hipe_final'

methods = [hipe_experiment,
           'rise',
           'center',
           'contrastive_excitation_backprop',
           'deconvnet',
           'excitation_backprop',
           'grad_cam',
           'gradient',
           'guided_backprop',
           'extremal_perturbation'
           ]

""" 
    ['rise',
    'center',
    'contrastive_excitation_backprop',
    'deconvnet',
    'excitation_backprop',
    'grad_cam',
    'gradient',
    'guided_backprop',
    'extremal_perturbation'
    ]
"""


class ProcessingError(Exception):

    def __init__(self, executor, experiment, model, image, label, class_id, image_size):
        super().__init__(f"Error processing {str(label):20s}")
        self.executor = executor
        self.experiment = experiment
        self.model = model
        self.image = image
        self.label = label
        self.class_id = class_id
        self.image_size = image_size


class ExperimentExecutor():

    def __init__(self, experiment, chunk=None, debug=0, seed=seed):
        print(experiment)
        self.experiment = experiment
        self.device = None
        self.model = None
        self.data = None
        self.loader = None
        self.insertion = 0
        self.deletion = 0
        self.time = 0
        self.num_ops = 0
        self.image_count = 0
        self.debug = debug
        self.seed = seed

        if self.experiment.arch == 'vgg16':
            self.gradcam_layer = 'features.29'  # relu before pool5
            self.saliency_layer = 'features.23'  # pool4
            self.contrast_layer = 'classifier.4'  # relu7
        elif self.experiment.arch == 'resnet50':
            self.gradcam_layer = 'layer4'
            self.saliency_layer = 'layer3'  # 'layer3.5'  # res4a
            self.contrast_layer = 'avgpool'  # pool before fc layer
        else:
            assert False

        if self.experiment.dataset == 'voc_2007':
            subset = 'test'
        elif self.experiment.dataset == 'coco':
            subset = 'val2014'
        else:
            assert False

        # Load the model.
        if self.experiment.method == "rise":
            input_size = (224, 224)
        else:
            input_size = 224
        transform = get_transform(size=input_size,
                                  dataset=self.experiment.dataset)

        self.data = get_dataset(name=self.experiment.dataset,
                                subset=subset,
                                transform=transform,
                                download=False,
                                limiter=lim)

        # Get subset of data. This is used for debugging and for
        # splitting computation on a cluster.
        if chunk is None:
            chunk = self.experiment.chunk

        if isinstance(chunk, dict):
            dataset_filter = chunk
            chunk = []
            if 'image_name' in dataset_filter:
                for i, name in enumerate(self.data.images):
                    if dataset_filter['image_name'] in name:
                        chunk.append(i)

            print(f"Filter selected {len(chunk)} image(s).")

        # Limit the chunk to the actual size of the dataset.
        if chunk is not None:
            chunk = list(set(range(len(self.data))).intersection(set(chunk)))

        # Extract the data subset.
        chunk = Subset(self.data, chunk) if chunk is not None else self.data

        # Get a data loader for the subset of data just selected.
        self.loader = DataLoader(chunk,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=self.data.collate)

        self.insertion = 0
        self.deletion = 0

        self.data_iterator = iter(self.loader)

    def _lazy_init(self):
        if self.device is not None:
            return

        self.device = get_device()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = get_model(
                arch=self.experiment.arch,
                dataset=self.experiment.dataset,
                convert_to_fully_convolutional=True,
                )

        # Some methods require patching the models further for
        # optimal performance.
        if self.experiment.arch == 'resnet50':
            if any([e in self.experiment.method for e in [
                'contrastive_excitation_backprop',
                'deconvnet',
                'excitation_backprop',
                'grad_cam',
                'gradient',
                'guided_backprop']]):

                self.model.avgpool = torch.nn.AvgPool2d((7, 7), stride=1)

            if 'excitation_backprop' in self.experiment.method:
                # Replace skip connection with EltwiseSum.
                self.model = update_resnet(self.model, debug=True)

        # Change model to eval modself.
        self.model.eval()

        # Move model to device.
        self.model.to(self.device)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader.dataset)

    def __next__(self):
        self._lazy_init()
        x, y = next(self.data_iterator)
        torch.manual_seed(self.seed)

        try:
            assert len(x) == 1
            x = x.to(self.device)
            class_ids = self.data.as_class_ids(y[0])
            image_size = self.data.as_image_size(y[0])

            results = {
                'insertion': {},
                'deletion':  {},
                'time':      {},
                'num_ops':   {}
                }
            info = {}
            rise_saliency = None
            num_ops = 0

            for class_id in class_ids:

                tic = time.process_time()

                if self.experiment.method == hipe_experiment:
                    saliency, num_ops = hierarchical_perturbation(self.model, x, class_id, resize=image_size, perturbation_type='mean')

                elif self.experiment.method == "center":
                    w, h = image_size
                    saliency = torch.zeros((1, 1, h, w))
                    saliency[:, :, int(h / 2), int(w / 2)] = 1

                elif self.experiment.method == "gradient":
                    saliency = gradient(
                            self.model, x, class_id,
                            resize=image_size,
                            smooth=0.02,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "deconvnet":
                    saliency = deconvnet(
                            self.model, x, class_id,
                            resize=image_size,
                            smooth=0.02,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "guided_backprop":
                    saliency = guided_backprop(
                            self.model, x, class_id,
                            resize=image_size,
                            smooth=0.02,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "grad_cam":
                    saliency = grad_cam(
                            self.model, x, class_id,
                            saliency_layer=self.gradcam_layer,
                            resize=image_size,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "excitation_backprop":
                    saliency = excitation_backprop(
                            self.model, x, class_id, self.saliency_layer,
                            resize=image_size,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "contrastive_excitation_backprop":
                    saliency = contrastive_excitation_backprop(
                            self.model, x, class_id,
                            saliency_layer=self.saliency_layer,
                            contrast_layer=self.contrast_layer,
                            resize=image_size,
                            get_backward_gradient=get_pointing_gradient
                            )

                elif self.experiment.method == "rise":
                    # For RISE, compute saliency map for all classes.
                    num_ops = 0
                    if rise_saliency is None:
                        num_ops = 8000
                        rise_saliency = rise(self.model, x, resize=image_size, seed=self.seed)
                    saliency = rise_saliency[:, class_id, :, :].unsqueeze(1)

                elif self.experiment.method == "extremal_perturbation":

                    if self.experiment.dataset == 'voc_2007':
                        areas = [0.025, 0.05, 0.1, 0.2]
                    else:
                        areas = [0.018, 0.025, 0.05, 0.1]

                    if self.experiment.boom:
                        raise RuntimeError("BOOM!")
                    num_ops = 800
                    mask, energy = elp.extremal_perturbation(
                            self.model, x, class_id,
                            areas=areas,
                            num_levels=8,
                            step=7,
                            sigma=7 * 3,
                            max_iter=800,
                            debug=self.debug > 0,
                            jitter=True,
                            smooth=0.09,
                            resize=image_size,
                            perturbation='fade',
                            reward_func=elp.simple_reward,
                            variant=elp.PRESERVE_VARIANT,
                            )

                    saliency = mask.sum(dim=0, keepdim=True)

                    info = {
                        'saliency': saliency,
                        'mask':     mask,
                        'areas':    areas,
                        'energy':   energy
                        }
                else:
                    assert False

                toc = time.process_time()

                info['saliency'] = saliency
                info['num_ops'] = num_ops

                base_model = get_model(
                        arch=self.experiment.arch,
                        dataset=self.experiment.dataset,
                        convert_to_fully_convolutional=True,
                        )

                base_model.eval()
                saliency = F.interpolate(saliency, (x.shape[-2], x.shape[-1]), mode='nearest').detach()

                insertion_metric = causal_metric(base_model,
                                                 saliency[0][0],
                                                 x,
                                                 class_id,
                                                 verbose=0,
                                                 mode='ins', save_as=self.experiment.name)

                ins_auc = auc(insertion_metric)

                deletion_metric = causal_metric(base_model,
                                                saliency[0][0],
                                                x,
                                                class_id,
                                                verbose=0,
                                                mode='del', save_as=self.experiment.name)

                del_auc = auc(deletion_metric)

                results['insertion'][class_id] = ins_auc
                results['deletion'][class_id] = del_auc
                results['time'][class_id] = toc - tic
                results['num_ops'][class_id] = num_ops

                self.image_count += 1

                print(self.experiment.name)
                print(results)

                if vis and 'coco' in self.experiment.dataset:

                    img = F.interpolate(x, saliency.shape[2:], mode='nearest')
                    image_name = self.data.as_image_name(y[0])
                    cls = classes[class_id]
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.title('')
                    imsc(saliency[0])
                    plt.subplot(1, 2, 2)
                    imsc(img[0])
                    plt.savefig('data/{}/pngs/'.format(series) + str(cls) + '_' + self.experiment.name + image_name)
                    plt.close()

            return results

        except Exception as ex:
            raise ProcessingError(
                    self, self.experiment, self.model, x, y, class_id, image_size) from ex

    def aggregate(self, results):

        for class_id, t in results['insertion'].items():
            self.insertion += t
        for class_id, t in results['deletion'].items():
            self.deletion += t
        for class_id, t in results['time'].items():
            self.time += t
        for class_id, t in results['num_ops'].items():
            self.num_ops += t

    def run(self, save=True):

        all_results = []
        for itr, results in enumerate(self):
            all_results.append(results)
            self.aggregate(results)
            print("[{}/{}]".format(itr + 1, len(self)))

        self.experiment.insertion = self.insertion / self.image_count
        self.experiment.deletion = self.deletion / self.image_count
        self.experiment.time = self.time / self.image_count
        self.experiment.num_ops = self.num_ops / self.image_count
        if save:
            self.experiment.save()
        return all_results

    def __str__(self):
        return (
            f"{self.experiment.method} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"insertion: {self.insertion}\n"
            f"{self.experiment.method} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"deletion: {self.deletion}\n"
            f"time: {self.time}"
            f"time: {self.num_ops}"
        )


class Experiment():

    def __init__(self,
                 series,
                 method,
                 arch,
                 dataset,
                 root='',
                 chunk=None,
                 boom=False):
        self.series = series
        self.root = root
        self.method = method
        self.arch = arch
        self.dataset = dataset
        self.chunk = chunk
        self.boom = boom
        self.insertion = float('NaN')
        self.deletion = float('NaN')
        self.num_ops = 0

    def __str__(self):
        return (
            f"{self.method},{self.arch},{self.dataset},"
            f"{self.insertion:.5f},{self.deletion:.5f},"
        )

    @property
    def name(self):
        return f"{self.method}-{self.arch}-{self.dataset}"

    @property
    def path(self):
        return os.path.join(self.root, self.name + '_cm_' + str(lim) + ".csv")

    def save(self):
        print(self.__str__() + "\n")
        with open(os.path.join(self.root, 'experiments' + '_cm_'+ str(lim) + '.csv'), 'a+') as f:
            f.write(self.__str__() + "\n")
        with open(os.path.join(self.root, self.name + '_cm_' + str(lim) + ".csv"), 'a+') as f:
            f.write(self.__str__() + "\n")

    def load(self):
        with open(self.path, "r") as f:
            data = f.read()
        method, arch, dataset, insertion, deletion, time, num_ops, _ = data.split(",")
        assert self.method == method
        assert self.arch == arch
        assert self.dataset == dataset
        self.insertion = float(insertion)
        self.deletion = float(deletion)
        self.time = float(time)
        self.num_ops = float(num_ops)

    def done(self):
        return os.path.exists(self.path)


experiments = []
xmkdir(series_dir)

for d in datasets:
    for a in archs:
        for m in methods:
            experiments.append(
                    Experiment(series=series,
                               method=m,
                               arch=a,
                               dataset=d,
                               chunk=chunk,
                               root=series_dir))

if __name__ == "__main__":
    for e in experiments:
        if e.done():
            e.load()
            continue
        ExperimentExecutor(e, debug=0).run()

