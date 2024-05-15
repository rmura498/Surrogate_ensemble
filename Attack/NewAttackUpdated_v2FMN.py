# @title Proposed Attack Code
import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
import gc
import numpy as np
from Utils.fmn_new_eps import FMN
from torch import nn
from adv_lib.attacks.projected_gradient_descent import pgd_linf


class newProposedFMN:

    def __init__(
        self,
        victim_model,
        ens_surrogates,
        attack_iterations,
        alpha,
        eps,
        pgd_iterations=10,
        loss="CW",
        device="cuda",
        sw=10, 
        lmb=0.5
    ):

        self.device = device
        self.victim_model = victim_model
        self.ens_surrogates = ens_surrogates
        self.attack_iterations = attack_iterations
        self.alpha = alpha
        self.eps = eps
        self.pgd_iterations = pgd_iterations
        self.sw=sw
        self.lmb=lmb
        self.loss_name = loss

        loss_functions = {"CW": CWLoss(), "CE": CrossEntropyLoss()}
        self.mse = nn.functional.mse_loss

        self.loss_fn = loss_functions[self.loss_name]

    def _compute_model_loss(self, model, advx, target, weights):

        model.eval()
        logits = model(advx)
        logits = logits.detach()
        pred_label = (logits).argmax()
        loss = self.loss_fn(logits, target)

        return loss, pred_label, logits

    def _pgd_cycle(self, weights, advx, target, image):
        
        advx = pgd_linf(ens_surrogates=self.ens_surrogates, weights=weights, inputs=advx, labels=target, eps=self.eps, targeted=True, steps=self.pgd_iterations,
             random_init=False, restarts=5, loss_function=self.loss_name, absolute_step_size=self.alpha)

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

        ensemble_outputs = torch.stack([model(advx) for model in ens_surrogates], dim=0)
        surrogate_sets = ensemble_outputs.detach().squeeze(dim=1)
        s = ensemble_outputs.sum(dim=0)

        mean_distance = torch.linalg.norm(victim_model(advx) - s, ord=2)


        return mean_distance, surrogate_sets

    def forward(self, image, true_label, target_label):

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        advx = torch.clone(image).unsqueeze(dim=0).to(self.device)


        loss_list = []
        weights_list = []
        dist_list = []
        mse_list = []
        self.surr_loss_list = []

        init_loss, pred_label, _ = self._compute_model_loss(
            self.victim_model, image.unsqueeze(dim=0), target_label, weights
        )

        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        loss_list.append(init_loss.detach().item())

        n_query = 0
        s_wind = self.sw
        lambt = torch.tensor(self.lmb, dtype=torch.double)

        model_drop_counter = torch.zeros(numb_surrogates).to(self.device)
        for n_step in range(self.attack_iterations):

            print(f"#Iteration - {n_step}#")

            if n_step == 0:

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

                    return n_query, loss_list, n_step, weights_list, mse_list, self.surr_loss_list

                print(f"Mean dist iter {n_step}:", dist.item())

                B = torch.clone(victim_logits).to(self.device)
                # A = torch.stack(surrogate_sets, dim=1)
                A = surrogate_sets.detach().clone().T
                B = torch.squeeze(B, 0).T


                # Ridge Regression
                w = self._pytorch_ridge_regression(A, B, lambt)
                weights = torch.clone(w).squeeze().to(self.device)
                weights_before = weights.clone().detach().to(self.device)

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

                    return n_query, loss_list, n_step, weights_list, mse_list, self.surr_loss_list

                print(
                    f"pred_label={pred_label.item()}, "
                    f"target={target_label.detach().item()}, queries={n_query} "
                    f"victmin loss={loss_victim.item()}"
                )

                
                ens_logit = torch.zeros_like(surrogate_sets[0])
                for i , out in enumerate(surrogate_sets):
                    ens_logit += (weights[i]*out)
                mse_value = self.mse(ens_logit, victim_logits).item() 
                print("Mse value:", mse_value)
                mse_list.append(mse_value)

                # newA = torch.stack(surrogate_sets, dim=1).to(self.device)
                newA = surrogate_sets.detach().clone().T
                newB = torch.squeeze(torch.clone(victim_logits).to(self.device), 0)

                if n_step == 1:
                    A = newA 
                    B = newB  
                
                #print(A.shape, newA.shape)
                if A.shape[0] == s_wind*newA.shape[0]:
                    
                    logit_size = newA.shape[0]
                    
                    A = A[logit_size:, :]
                    B = B[logit_size:]
                    A = torch.cat([A, newA], dim=0).to(self.device)  
                    B = torch.cat([B, newB], dim=0).to(self.device)
                    #print(A.shape, B.shape)

                else:

                    A = torch.cat([A, newA], dim=0).to(self.device)  
                    B = torch.cat([B, newB], dim=0).to(self.device)  

                w = self._pytorch_ridge_regression(A, B, lambt)
                weights = torch.clone(w).squeeze().to(self.device)
                #print(weights)
                #if n_step > 1:  
                #    weights = 1*last_weights + weights / torch.norm(weights[:-1], p=2)
                    # intercept = torch.abs(weights[-1]+0.5)
                #last_weights = weights


        return n_query, loss_list, self.attack_iterations, weights_list, mse_list, self.surr_loss_list
