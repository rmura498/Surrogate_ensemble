import numbers
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from adv_lib.utils.projections import clamp_
from adv_lib.utils.visdom_logger import VisdomLogger
from Utils.CW_loss import CWLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


def pgd_linf(ens_surrogates: list,
             weights: list,
             inputs: Tensor,
             labels: Tensor,
             eps: Union[float, Tensor],
             targeted: bool = False,
             steps: int = 40,
             random_init: bool = True,
             restarts: int = 5,
             loss_function: str = 'ce',
             relative_step_size: float = 0.01 / 0.3,
             absolute_step_size: Optional[float] = None,
             callback: Optional[VisdomLogger] = None) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if isinstance(eps, numbers.Real):
        eps = torch.full_like(adv_found, eps, dtype=inputs.dtype)

    pgd_attack = partial(_pgd_linf, ens_surrogates=ens_surrogates, weights=weights, targeted=targeted, steps=steps, random_init=random_init,
                         loss_function=loss_function, relative_step_size=relative_step_size,
                         absolute_step_size=absolute_step_size)

    for i in range(restarts):
        
        adv_found_run, adv_inputs_run = pgd_attack(inputs=inputs[~adv_found], labels=labels[~adv_found],
                                                   eps=eps[~adv_found])
        adv_inputs[~adv_found] = adv_inputs_run
        adv_found[~adv_found] = adv_found_run

        if callback:
            callback.line('success', i + 1, adv_found.float().mean())

        if adv_found.all():
            break

    return adv_inputs


def _pgd_linf(ens_surrogates: list,
              weights: list,
              inputs: Tensor,
              labels: Tensor,
              eps: Tensor,
              targeted: bool = False,
              steps: int = 40,
              random_init: bool = True,
              loss_function: str = 'ce',
              relative_step_size: float = 0.01 / 0.3,
              absolute_step_size: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    _loss_functions = {
        'ce': (partial(F.cross_entropy, reduction='none'), 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
        'cw': (partial(CWLoss()), 1)
    }


    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    lower, upper = torch.maximum(-inputs, -batch_view(eps)), torch.minimum(1 - inputs, batch_view(eps))

    loss_func, multiplier = _loss_functions[loss_function.lower()]

    if absolute_step_size is not None:
        step_size = eps
    else:
        step_size = eps * relative_step_size

    if targeted:
        step_size *= 1

    delta = torch.zeros_like(inputs, requires_grad=True)
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    if random_init:
        delta.data.uniform_(-1, 1).mul_(batch_view(eps))
        clamp_(delta, lower=lower, upper=upper)

    optimizer = Adam([delta], lr=step_size.mean().item())
    #scheduler = CosineAnnealingLR(optimizer, T_max=steps)


    for i in range(steps):
        optimizer.zero_grad()

        adv_inputs = inputs + delta
        # logits = torch.stack([weights[i] * model(adv_inputs) for i, model in enumerate(ens_surrogates)], dim=0).sum(dim=0)
        ensemble_outputs = torch.stack([model(adv_inputs) for model in ens_surrogates], dim=0)
        # print(f"weights shape: {weights.shape}")
        # print(f"Ensemble outputs shape: {ensemble_outputs.shape}")
        logits = (weights[:len(ensemble_outputs)].view(-1, 1, 1) * ensemble_outputs).sum(dim=0)

        if i == 0 and loss_function.lower() in ['dl', 'dlr']:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, labels)
        loss.sum().backward()
        # delta_grad = grad(loss.sum(), delta, only_inputs=True)[0].sign_().mul_(batch_view(step_size))
        delta.grad.data.sign_().mul_(batch_view(step_size))
        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        best_adv = torch.where(batch_view(is_adv), adv_inputs.detach(), best_adv)
        adv_found.logical_or_(is_adv)
        optimizer.step()                
        clamp_(delta, lower=lower, upper=upper)
        #scheduler.step()

    return adv_found, best_adv
