import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss


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

    def _pgd_cycle(self, image, weights, advx, target):

        numb_surrogates = len(self.ens_surrogates)

        for i in range(self.pgd_iterations):
            advx.requires_grad_()

            outputs = [model(advx) for i, model in enumerate(self.ens_surrogates)]
            loss = sum([weights[idx] * self.loss_fn(outputs[idx], target) for idx in range(numb_surrogates)])
            loss.backward()

            with torch.no_grad():
                grad = advx.grad
                advx = advx - self.alpha * torch.sign(grad)  # perturb x
                advx = (image + (advx - image).clamp(min=-self.eps, max=self.eps)).clamp(0, 1)

        return advx

    def _mean_logits_distance(self, advx, weights, victim_model, ens_surrogates):
        surrogate_sets = [model(advx).detach().squeeze(dim=0) for model in ens_surrogates]
        outputs = [torch.norm((victim_model(advx) - weights[i] * surr_log.unsqueeze(dim=0)), p=2).item() for
                   i, surr_log in enumerate(surrogate_sets)]
        mean_distance = torch.mean(torch.tensor(outputs))

        return mean_distance, surrogate_sets

    def forward(self, image, true_label, target_label):

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        advx = torch.clone(image).unsqueeze(dim=0).detach().to(self.device)

        loss_list = []
        weights_list = []
        weights_list.append(weights.cpu().numpy().tolist())

        init_loss, pred_label, _ = self._compute_model_loss(self.victim_model, image.unsqueeze(dim=0), target_label)

        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        loss_list.append(init_loss.detach().item())

        n_query = 0

        for n in range(5):
            advx = self._pgd_cycle(image, weights, advx, target_label)
            loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label)
            if pred_label == target_label:
                print(f"Success pred_label={pred_label.item()}, "
                        f"target={target_label.detach().item()}, queries={n_query},"
                        f"victmin loss={loss_victim.item()}")
                return 0, loss_list, 'ASR:1', weights_list

        return 40, loss_list, 'ASR:0', weights_list
