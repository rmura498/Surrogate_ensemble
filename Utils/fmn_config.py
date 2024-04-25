from Utils.loss import LL, DLR, CE
from torch.optim import SGD, Adam, Adamax
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

LOSSES = {
    'LL': LL,
    'DLR': DLR,
    'CE': CE
}

OPTIMIZERS = {
    'SGD': SGD,
    'Adam': Adam,
    'Adamax': Adamax
}

SCHEDULERS = {
    'CALR': CosineAnnealingLR,
    'RLROP': ReduceLROnPlateau
}


def print_fmn_configs():
    print("\nLosses:")
    for loss in LOSSES.keys():
        print(f"\t{loss}")

    print("Optimizers:")
    for optimizer in OPTIMIZERS.keys():
        print(f"\t{optimizer}")

    print("Schedulers:")
    for scheduler in SCHEDULERS.keys():
        print(f"\t{scheduler}")