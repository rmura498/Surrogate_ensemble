import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import Utils.imagenet_1k
from Utils.imagenet_1k import normalize_model

mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)

def load_dataset(dataset_name='imagenet'):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('./Models/data',
                                             train=False,
                                             download=True,
                                             transform=transform)
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./Models/data',
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
    elif dataset_name == 'imagenet':
        dataset = Utils.imagenet_1k.ImageNet1K(dataset_root='./Models/data/imagenet1000',
                                               transform=transform)
    else:
        raise NotImplementedError("Unknown dataset")

    return dataset


def load_model(model_name, device, normalization=True):
    """Load the model according to the idx in list model_names

    Args:
        model_name (str): the name of model, chosen from the following list
        model_names = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1', \
            'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5', \
            'mobilenet_v2', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet1_0', 'mnasnet0_5']
    Returns:
        model (torchvision.models): the loaded model
    """
    model = getattr(models, model_name)(pretrained=True).to(device).eval()
    if normalization:
        model = normalize_model(model, mu, sigma)
    return model


def load_surrogates(surrogate_list, device):
    ens_surrogates = []

    for model_name in surrogate_list:
        print(f"load: {model_name}")
        ens_surrogates.append(load_model(model_name, device))

    return ens_surrogates
