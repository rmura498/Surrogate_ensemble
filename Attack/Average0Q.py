import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
from adv_lib.attacks.projected_gradient_descent import pgd_linf


class Average0():
    def __init__(self, victim_model, ens_surrogates, attack_iterations,
                 alpha, lr, eps, pgd_iterations=10, loss='CW', device='cuda'):

        self.device = device
        self.victim_model = victim_model
        self.ens_surrogates = ens_surrogates
        self.attack_iterations = attack_iterations
        self.alpha = alpha
        self.lr = lr
        self.eps = eps
        self.pgd_iterations = pgd_iterations

        loss_functions = {'CW': CWLoss(),
                          'CE': CrossEntropyLoss()}

        self.loss_fn = loss_functions[loss]
        self.victim_model.to(self.device)

    def _compute_model_loss(self, model, advx, target):

        model.eval()
        logits = model(advx)
        logits = logits.detach()
        pred_label = logits.argmax()
        loss = self.loss_fn(logits, target)

        return loss, pred_label, logits

    # def _pgd_cycle(self, weights, advx, target, image):
    def _pgd_cycle(self, image, weights, target):
        
        advx = pgd_linf(ens_surrogates=self.ens_surrogates, weights=weights, inputs=image, labels=target, eps=self.eps, targeted=True, steps=self.pgd_iterations,
             random_init=False, restarts=1, loss_function='cw', absolute_step_size=self.alpha)
        
        _norm = (advx-image).data.flatten(1).norm(p=torch.inf, dim=1).median().item()
        print(f"[BB attack info] norm after PGD: {_norm}")
        
        return advx

    def forward(self, image, true_label, target_label):

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        image = image.unsqueeze(0).to(self.device)
        advx = torch.clone(image).unsqueeze(dim=0).detach().to(self.device)

        loss_list = []
        weights_list = []
        weights_list.append(weights.cpu().numpy().tolist())

        init_loss, pred_label, _ = self._compute_model_loss(self.victim_model, image, target_label)

        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        loss_list.append(init_loss.detach().item())

        n_query = 0
        for i in range(5):

            # advx = self._pgd_cycle(image=image, weights=weights, advx=advx, target=target_label)
            advx = self._pgd_cycle(image=image, weights=weights, target=target_label)
            loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label)
            if pred_label == target_label:
                print(f"Success pred_label={pred_label.item()}, "
                        f"target={target_label.detach().item()}, queries={n_query},"
                        f"victmin loss={loss_victim.item()}")
                return 0, loss_list, 'ASR:1', weights_list

        return 40, loss_list, 'ASR:0', weights_list
