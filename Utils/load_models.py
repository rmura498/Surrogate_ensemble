from collections import defaultdict
import torchvision.models as models
from pathlib import Path
import csv
def load_imagenet_1000(dataset_root = "imagenet1000"):
    """
    Dataset downoaded form kaggle
    https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set
    Resized from 299x299 to 224x224
    Args:
        dataset_root (str): root folder of dataset
    Returns:
        img_paths (list of strs): the paths of images
        gt_labels (list of ints): the ground truth label of images
        tgt_labels (list of ints): the target label of images
    """
    dataset_root = Path(dataset_root)
    img_paths = list(sorted(dataset_root.glob('*.png')))
    gt_dict = defaultdict(int)
    tgt_dict = defaultdict(int)
    with open(dataset_root / "images.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gt_dict[row['ImageId']] = int(row['TrueLabel'])
            tgt_dict[row['ImageId']] = int(row['TargetClass'])
    gt_labels = [gt_dict[key] - 1 for key in sorted(gt_dict)] # zero indexed
    tgt_labels = [tgt_dict[key] - 1 for key in sorted(tgt_dict)] # zero indexed
    return img_paths, gt_labels, tgt_labels

def load_model(model_name, device):
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
    return model
