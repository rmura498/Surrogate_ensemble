# @title Proposed Attack Code
import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
import gc
import numpy as np


class newProposed:

    def __init__(
        self,
        victim_model,
        ens_surrogates,
        attack_iterations,
        alpha,
        lr,
        eps,
        pgd_iterations=10,
        loss="CW",
        device="cuda",
    ):

        self.device = device
        self.victim_model = victim_model
        self.ens_surrogates = ens_surrogates
        self.attack_iterations = attack_iterations
        self.alpha = alpha
        self.lr = lr
        self.eps = eps
        self.pgd_iterations = pgd_iterations

        loss_functions = {"CW": CWLoss(), "CE": CrossEntropyLoss()}

        self.loss_fn = loss_functions[loss]

    def _compute_model_loss(self, model, advx, target, weights):

        model.eval()
        logits = model(advx)
        logits = logits.detach()
        pred_label = (torch.softmax(logits, dim=1)).argmax()
        loss = self.loss_fn(torch.softmax(logits, dim=1), target)

        return loss, pred_label, logits

    def _pgd_cycle(self, weights, advx, target, image):

        numb_surrogates = len(self.ens_surrogates)

        for i in range(self.pgd_iterations):
            advx.requires_grad_()

            outputs = [torch.softmax(model(advx), dim=1) + weights[-1] for i, model in enumerate(self.ens_surrogates)]
            loss = sum([weights[idx] * self.loss_fn(outputs[idx], target) for idx in range(numb_surrogates)])

            loss.backward()

            with torch.no_grad():
                grad = advx.grad
                advx = advx - self.alpha * torch.sign(grad)  # perturb x
                advx = (image + (advx - image).clamp(min=-self.eps, max=self.eps)).clamp(0, 1)

        return advx

    def _pytorch_ridge_regression(self, X, y, alpha, fit_intercept=True):

        n_samples, n_features = X.shape[0], X.shape[1]

        if fit_intercept:
            A = torch.zeros(size=(n_samples + n_features, n_features + 1))
            A[:n_samples, :-1] = X
            A[:n_samples, -1] = 1
            A[n_samples:, :n_features] = (alpha**0.5) * torch.eye(n_features)
            A[n_samples:, -1] = 0

        else:
            A = torch.zeros(size=(n_samples + n_features, n_features))
            A[:n_samples, :] = X
            A[n_samples:, :n_features] = (alpha**0.5) * torch.eye(n_features)

        b = torch.zeros(size=(n_samples + n_features,))
        b[:n_samples] = y
        w = torch.linalg.lstsq(A, b).solution

        return w

    def _mean_logits_distance(self, advx, weights, victim_model, ens_surrogates):

        surrogate_sets = [torch.softmax(model(advx), dim=1).detach().squeeze(dim=0) for model in ens_surrogates]
        s = sum([weights[i] * surr_log.unsqueeze(dim=0) for i, surr_log in enumerate(surrogate_sets)])
        mean_distance = torch.norm((torch.softmax(victim_model(advx), dim=1) - s - weights[-1]), p=2)

        return mean_distance, surrogate_sets

    def forward(self, image, true_label, target_label):

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        advx = torch.clone(image).unsqueeze(dim=0).detach().to(self.device)

        alpha = 0.1
        alphat = torch.tensor(alpha, dtype=torch.double)

        loss_list = []
        weights_list = []
        dist_list = []

        init_loss, pred_label, _ = self._compute_model_loss(
            self.victim_model, image.unsqueeze(dim=0), target_label, weights
        )

        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        loss_list.append(init_loss.detach().item())

        n_query = 0

        for n_step in range(self.attack_iterations):

            print(f"#Iteration - {n_step}#")

            if n_step == 0:

                # print("Step:", step)
                # print("weights:",weights)
                advx = self._pgd_cycle(weights, advx, target_label, image)

                dist, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)
                dist_list.append(dist.detach().cpu().item())
                loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label, weights)
                n_query += 1
                loss_list.append(loss_victim.detach().item())
                weights_list.append(weights.cpu().numpy().tolist())

                if pred_label == target_label:
                    print(
                        f"Success pred_label={pred_label.item()}, "
                        f"target={target_label.detach().item()}, queries={n_query}, "
                        f"victmin loss={loss_victim.item()}"
                    )

                    return n_query, loss_list, n_step, weights_list

                print(f"Mean dist iter {n_step}:", dist.item())

                B = torch.clone(victim_logits).to(self.device)
                A = torch.stack(surrogate_sets, dim=1)
                B = torch.squeeze(B, 0).T  # i'm transposing B just because the 'victim_logit' tensor is 1x1000

                # Ridge Regression
                w = self._pytorch_ridge_regression(A, B, alphat)
                weights = torch.clone(w).squeeze().to(self.device)

            else:

                advx = self._pgd_cycle(weights, advx, target_label, image)

                loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label, weights)
                n_query += 1
                loss_list.append(loss_victim.item())
                weights_list.append(weights.cpu().numpy().tolist())

                dist, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)
                dist_list.append(dist.detach().cpu().item())
                print(f"Mean dist iter {n_step}:", dist.item())

                if pred_label == target_label:
                    print(
                        f"Success pred_label={pred_label.item()}, "
                        f"target={target_label.detach().item()}, queries={n_query}, "
                        f"victmin loss={loss_victim.item()}"
                    )

                    return n_query, loss_list, n_step, weights_list

                print(
                    f"pred_label={pred_label.item()}, "
                    f"target={target_label.detach().item()}, queries={n_query} "
                    f"victmin loss={loss_victim.item()}"
                )


                newA = torch.stack(surrogate_sets, dim=1).to(self.device)
                newB = torch.squeeze(torch.clone(victim_logits).to(self.device), 0)

                A = torch.cat([A, newA], dim=0).to(self.device)  # here A is computed at the previous iteration
                B = torch.cat([B, newB], dim=0).to(self.device)  # i'm transposing newB just because the tensor is 1x1000

                w = self._pytorch_ridge_regression(A, B, alphat)
                weights = torch.clone(w).squeeze().to(self.device)


        return n_query, loss_list, self.attack_iterations, weights_list 