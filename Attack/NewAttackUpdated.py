import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


class newProposed():

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

    def _compute_model_loss(self, model, advx, target):

        model.eval()
        logits = model(advx)
        logits = logits.detach()
        pred_label = logits.argmax()
        loss = self.loss_fn(logits, target)

        return loss, pred_label, logits

    def _pgd_cycle(self, weights, advx, target, image, optim=None):

        numb_surrogates = len(self.ens_surrogates)
        #optim.zero_grad()

        for i in range(self.pgd_iterations):
            advx.requires_grad_()

            outputs = [weights[i] * model(advx) for i, model in enumerate(self.ens_surrogates)]
            loss = sum([weights[idx] * self.loss_fn(outputs[idx], target) for idx in range(numb_surrogates)])

            loss.backward()
            #optim.step()
            #step = optim.param_groups[0]['lr']
            with torch.no_grad():
                grad = advx.grad
                advx = advx - self.alpha * torch.sign(grad)  # perturb x
                advx = (image + (advx - image).clamp(min=-self.eps,max=self.eps)).clamp(0,1)

        return advx, loss

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


        #optimizer = SGD([advx], lr=self.alpha)
        #scheduler = MultiStepLR(optimizer, milestones=[40,50], gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.75)

        loss_list = []
        weights_list = []

        init_loss, pred_label, _ = self._compute_model_loss(self.victim_model, image.unsqueeze(dim=0), target_label)

        print("True label", true_label.item())
        print("pred label", pred_label.item())
        print("inital victim loss", init_loss.item())
        loss_list.append(init_loss.detach().item())

        n_query = 0

        for n_step in range(self.attack_iterations):

            print(f'#Iteration - {n_step}#')

            if n_step == 0:

                #step = optimizer.param_groups[0]['lr']
                #print("Step:", step)
                #print("weights:",weights)
                advx, loss = self._pgd_cycle(weights, advx, target_label, image)


                dist, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)
                loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label)
                n_query += 1
                loss_list.append(loss_victim.detach().item())
                weights_list.append(weights.cpu().numpy().tolist())

                if pred_label == target_label:
                    print(f"Success pred_label={pred_label.item()}, "
                            f"target={target_label.detach().item()}, queries={n_query}, "
                            f"victmin loss={loss_victim.item()}")
                    return n_query, loss_list, n_step, weights_list
                print(f"Mean dist iter {n_step}:",dist.item())
                #scheduler.step(loss_victim)
                #scheduler.step(dist)

                B = torch.clone(victim_logits).to(self.device)
                A = torch.stack(surrogate_sets, dim=1)
                B = B.T #i'm transposing B just because the 'victim_logit' tensor is 1x1000
                #print(A.shape)
                #print(B.shape)
                solution = torch.linalg.lstsq(A, B).solution
                weights = torch.clone(solution).squeeze().to(self.device)

                dist, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)
                print(f"Mean dist iter after first weights {n_step}:",dist.item())

            else:

                #step = optimizer.param_groups[0]['lr']
                #print("Step:", step)
                #print("weights:",weights)
                advx, loss = self._pgd_cycle(weights, advx, target_label, image)

                loss_victim, pred_label, victim_logits = self._compute_model_loss(self.victim_model, advx, target_label)
                n_query += 1
                loss_list.append(loss_victim.item())
                weights_list.append(weights.cpu().numpy().tolist())

                _, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)

                if pred_label == target_label:
                    print(f"Success pred_label={pred_label.item()}, "
                            f"target={target_label.detach().item()}, queries={n_query}, "
                            f"victmin loss={loss_victim.item()}")
                    return n_query, loss_list, n_step, weights_list

                print(f"pred_label={pred_label.item()}, "
                        f"target={target_label.detach().item()}, queries={n_query} "
                        f"victmin loss={loss_victim.item()}")
                print("Surr Loss", loss.item())

                #scheduler.step(loss)
                #scheduler.step(dist)

                newA = torch.stack(surrogate_sets, dim=1).to(self.device)
                newB = torch.clone(victim_logits).to(self.device)
                A = torch.cat([A, newA], dim=0).to(self.device) # here A is computed at the previous iteration
                B = torch.cat([B, newB.T], dim=0).to(self.device) #i'm transposing newB just because the tensor is 1x1000

                #print("A shape:",A.shape)
                #print(A.T.shape)
                #print(B.shape)

                lamb = 1
                I = lamb*torch.eye(A.shape[1]).to(self.device)
                #print(I.shape)

                X = ((A.T @ A) + I).to(self.device)
                #print("X", X.shape)

                Y = (A.T @ B).to(self.device)
                #print("Y", Y.shape)

                solution = torch.linalg.lstsq(X, Y).solution
                weights = torch.clone(solution).squeeze().to(self.device)

                solution1 = torch.linalg.lstsq(newA, newB.T).solution
                weights1 = torch.clone(solution1).squeeze().to(self.device)

                dist1, surrogate_sets = self._mean_logits_distance(advx, weights1, self.victim_model, self.ens_surrogates)
                print(f"Mean dist first sol {n_step}:",dist1.item())

                dist, surrogate_sets = self._mean_logits_distance(advx, weights, self.victim_model, self.ens_surrogates)
                print(f"Mean dist improved sol {n_step}:",dist.item())



        return n_query, loss_list, self.attack_iterations, weights_list