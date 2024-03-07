import torch
from config import SURROGATE_NAMES


def compute_alignment(image, victim_model, ens_surrogates, loss_fn, target):
    alignment_dict = {}
    image = image.unsqueeze(dim=0)
    image.requires_grad_()
    vic_outputs = victim_model(image)
    lossv = loss_fn(vic_outputs, target)
    lossv.backward()
    with torch.no_grad():
        vic_grad = image.grad.detach()
        g1 = torch.clone(torch.flatten(vic_grad))

    for i, model in enumerate(ens_surrogates):
        image.grad.data.zero_()
        outputs = model(image)
        loss = loss_fn(outputs, target)
        loss.backward()
        with torch.no_grad():
            surr_grad = image.grad.detach()
            g2 = torch.clone(torch.flatten(surr_grad))
            alignment = torch.dot(g1, g2) / torch.norm(g1, p=2) * torch.norm(g2, p=2)
            alignment_dict[SURROGATE_NAMES[i]] = alignment.item()

    return alignment_dict
