# @title Proposed Attack Code
import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
import gc
import numpy as np
from torch import nn



class newProposedv1:

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

        loss_functions = {"CW": CWLoss(), "CE": CrossEntropyLoss()}
        self.mse = nn.functional.mse_loss

        self.loss_fn = loss_functions[loss]

    def _compute_model_loss(self, model, advx, target, weights):

        model.eval()
        logits = model(advx)
        logits = logits.detach()
        pred_label = (logits).argmax()
        loss = self.loss_fn(logits, target)

        return loss, pred_label, logits

    def _pgd_cycle(self, weights, advx, target, image):

        numb_surrogates = len(self.ens_surrogates)

        for i in range(self.pgd_iterations):
            advx.requires_grad_()

            outputs = [weights[i] *model(advx) for i, model in enumerate(self.ens_surrogates)]
            loss = sum([self.loss_fn(outputs[idx], target) for idx in range(numb_surrogates)])
            
            #print(loss)
            loss.backward()


            with torch.no_grad():
                grad = advx.grad
                advx = advx - self.alpha * torch.sign(grad)  # perturb x
                advx = (image + (advx - image).clamp(min=-self.eps, max=self.eps)).clamp(0, 1)
        self.surr_loss_list.append([self.loss_fn(outputs[idx], target).item() for idx in range(numb_surrogates)])
        self.surr_loss_list.append([loss.item()])
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

        #surrogate_sets = [weights[i]*model(advx).detach().squeeze(dim=0) for i, model in enumerate(ens_surrogates)]
        surrogate_sets = [model(advx).detach().squeeze(dim=0) for i, model in enumerate(ens_surrogates)]
        s = sum([surr_log.unsqueeze(dim=0) for i, surr_log in enumerate(surrogate_sets)])
        mean_distance = torch.norm(victim_model(advx) - s, p=2)

        return mean_distance, surrogate_sets

    def forward(self, image, true_label, target_label):

        numb_surrogates = len(self.ens_surrogates)
        weights = torch.ones(numb_surrogates).to(self.device) / numb_surrogates
        advx = torch.clone(image).unsqueeze(dim=0).detach().to(self.device)


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
                A = torch.stack(surrogate_sets, dim=1)
                B = torch.squeeze(B, 0).T 

                # Ridge Regression
                w = self._pytorch_ridge_regression(A, B, lambt)
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

                    return n_query, loss_list, n_step, weights_list, mse_list, self.surr_loss_list

                print(
                    f"pred_label={pred_label.item()}, "
                    f"target={target_label.detach().item()}, queries={n_query} "
                    f"victmin loss={loss_victim.item()}"
                )

                
                mse_value = [self.mse(surr_log, victim_logits).item() for surr_log in surrogate_sets]
                #print(mse_value)
                print("Average value1:", sum(mse_value))
                mse_list.append(sum(mse_value))

                newA = torch.stack(surrogate_sets, dim=1).to(self.device)
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


        return n_query, loss_list, self.attack_iterations, weights_list, mse_list, self.surr_loss_list
