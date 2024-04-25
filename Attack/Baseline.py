import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss


class Baseline():
    def __init__(self, victim_model, ens_surrogates, attack_iterations,
                 alpha, lr, eps, pgd_iterations=10, loss='CW', device='cuda', norm = 'Linf'):

        self.device = device
        self.victim_model = victim_model
        self.ens_surrogates = ens_surrogates
        self.attack_iterations = attack_iterations
        self.alpha = alpha
        self.lr = lr
        self.eps = eps
        self.pgd_iterations = pgd_iterations

        norm_to_num = {
            'Linf': float('inf'),
            'L2': 2,
            }

        self.norm = norm_to_num[norm]

        loss_functions = {'CW': CWLoss(),
                          'CE': CrossEntropyLoss()}

        self.loss_fn = loss_functions[loss]

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
                if self.norm == float('inf'):
                    advx = (image + (advx - image).clamp(min=-self.eps, max=self.eps)).clamp(0, 1)
                
                elif self.norm == 2:
                    pert = advx - image
                    l2_norm = torch.norm(pert, p=self.norm)
                    if l2_norm > eps:
                        pert = pert / l2_norm * self.eps
                        adv = (im + pert).clamp(0,1)

        return advx

    def _mean_logits_distance(self, advx, weights, victim_model, ens_surrogates):
        surrogate_sets = [model(advx).detach().squeeze(dim=0) for model in ens_surrogates]
        outputs = [torch.norm((victim_model(advx) - weights[i] * surr_log.unsqueeze(dim=0)), p=2).item() for
                   i, surr_log in enumerate(surrogate_sets)]
        mean_distance = torch.mean(torch.tensor(outputs))

        return mean_distance

    def forward(self, image, true_label, target_label):

        n_query = 0

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        advx = torch.clone(image).unsqueeze(dim=0).detach().to(self.device)

        idx_w = 0
        last_idx = 0
        v_loss_list = []
        weights_list = []

        init_loss, pred_label, _ = self._compute_model_loss(self.victim_model, image.unsqueeze(dim=0), target_label)
        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        v_loss_list.append(init_loss.detach().item())

        for n_step in range(self.attack_iterations // 2):
            print(f"Step: {n_step}")

            if n_step == 0:

                advx = self._pgd_cycle(image, weights, advx, target_label)

                loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label)
                v_loss_list.append(loss_victim.detach().item())
                n_query += 1


                weights_list.append(weights.cpu().numpy().tolist())
                if pred_label == target_label:
                    print(f"Success: pred={pred_label} - target={target_label} - query={n_query}")
                    return n_query, v_loss_list, n_step, weights_list

            else:
                # optimize w and adv with w = w + delta
                weights_plus = torch.clone(weights).to(self.device)
                weights_plus[idx_w] = weights_plus[idx_w] + self.lr

                advx_plus = self._pgd_cycle(image, weights_plus, advx, target_label)
                loss_plus, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx_plus,
                                                                                target_label)
                n_query += 1

                if pred_label == target_label:
                    print(
                        f"Success (plus): pred={pred_label} - target={target_label}, query:{n_query}, victim loss={loss_plus}")
                    v_loss_list.append(loss_plus.detach().item())
                    weights_list.append(weights_plus.cpu().numpy().tolist())
                    return n_query, v_loss_list, n_step, weights_list

                # optimize w and adv with w = w - delta
                weights_minus = torch.clone(weights).to(self.device)
                weights_minus[idx_w] = weights_minus[idx_w] - self.lr

                advx_minus = self._pgd_cycle(image, weights_minus, advx, target_label)
                loss_minus, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx_minus,
                                                                                 target_label)
                n_query += 1

                if pred_label == target_label:
                    print(f"Success (plus): pred={pred_label} - target={target_label}, "
                          f"query:{n_query}, victim loss={loss_plus}")
                    v_loss_list.append(loss_minus.detach().item())
                    weights_list.append(weights_minus.cpu().numpy().tolist())
                    return n_query, v_loss_list, n_step, weights_list

                # update weight and adversarial sample x using l+, l-, w+, w-, x+, x-
                if loss_plus < loss_minus:
                    loss = loss_plus
                    weights = torch.clone(weights_plus).to(self.device)
                    advx = torch.clone(advx_plus).detach().to(self.device)
                    last_idx = idx_w
                    v_loss_list.append(loss.detach().item())
                    weights_list.append(weights.cpu().numpy().tolist())

                else:
                    loss = loss_minus
                    weights = torch.clone(weights_minus).to(self.device)
                    advx = torch.clone(advx_minus).detach().to(self.device)
                    last_idx = idx_w
                    v_loss_list.append(loss.detach().item())
                    weights_list.append(weights.cpu().numpy().tolist())

                if n_query > 5 and last_idx == idx_w:
                    self.lr /= 2
                idx_w = (idx_w + 1) % numb_surrogates

                print(f"pred_label={pred_label.item()}, "
                      f"target={target_label.detach().item()}, queries={n_query} "
                      f"victmin loss={loss_victim.item()}")

        return n_query, v_loss_list, self.attack_iterations, weights_list
