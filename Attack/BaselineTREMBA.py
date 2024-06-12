import torch
from Utils.CW_loss import CWLoss
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from Utils.TREMBA_Utils import FCN


class BaselineTREMBA:
    def __init__(self, function, encoder_weight, decoder_weight, victim_model, attack_iterations,
                 alpha, lr, eps, pgd_iterations=10, loss='CW', device='cuda', norm='Linf'):

        self.device = device
        self.attack_iterations = attack_iterations
        self.alpha = alpha
        self.lr = lr
        self.eps = eps
        self.pgd_iterations = pgd_iterations
        self.encoder = FCN.Imagenet_Encoder()
        self.decoder = FCN.Imagenet_Decoder()
        self.function = function
        self.victim_model = victim_model

        self.loss_list = []

        norm_to_num = {
            'Linf': float('inf'),
            'L2': 2,
            }

        self.norm = norm_to_num[norm]

        loss_functions = {'CW': CWLoss(),
                          'CE': CrossEntropyLoss()}

        self.loss_fn = loss_functions[loss]

        self.encoder.load_state_dict(encoder_weight)
        self.decoder.load_state_dict(decoder_weight)
        self.encoder.to(self.device)
        self.decoder.to(self.device)


    def forward(self, images, labels, state):
        images = images.to(self.device)
        labels = int(labels)
        logits = self.victim_model(images)
        correct = torch.argmax(logits, dim=1) == labels
        if correct:
            torch.cuda.empty_cache()
            if state['target']:
                labels = state['target_class']
            with torch.no_grad():
                success, adv, query, n_iter = self.embed_ba(images, labels, state)

            self.function.new_counter()
            return query, self.loss_list, n_iter

    def embed_ba(self, image, label, config, latent=None):
        if latent is None:
            latent = self.encoder(image.unsqueeze(0)).squeeze().view(-1)
        momentum = torch.zeros_like(latent)
        dimension = len(latent)
        noise = torch.empty((dimension, config['sample_size']), device=self.device)
        origin_image = image.clone()
        last_loss = []
        lr = config['lr']
        for n_iter in range(config['num_iters'] + 1):
            perturbation = torch.clamp(self.decoder(latent.unsqueeze(0)).squeeze(0) * config['epsilon'], -config['epsilon'],
                                       config['epsilon'])
            logit, loss = self.function(torch.clamp(image + perturbation, 0, 1), label)
            self.loss_list.append(loss.detach().cpu().item())
            if config['target']:
                success = torch.argmax(logit, dim=1) == label
            else:
                success = torch.argmax(logit, dim=1) != label
            last_loss.append(loss.item())

            if self.function.current_counts > 50000:
                break

            if bool(success.item()):

                return True, torch.clamp(image + perturbation, 0, 1), n_iter + 1, n_iter

            nn.init.normal_(noise)
            # noise[:, config['sample_size'] // 2:] = -noise[:, :config['sample_size'] // 2]
            latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1) * config['sigma']
            perturbations = torch.clamp(self.decoder(latents) * config['epsilon'], -config['epsilon'], config['epsilon'])
            _, losses = self.function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

            grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

            if n_iter % config['log_interval'] == 0 and config['print_log']:
                print("iteration: {} loss: {}, l2_deviation {}".format(n_iter, float(loss.item()),
                                                                       float(torch.norm(perturbation))))
            self.loss_list.append(loss.item())
            momentum = config['momentum'] * momentum + (1 - config['momentum']) * grad

            latent = latent - lr * momentum

            last_loss = last_loss[-config['plateau_length']:]
            if ((last_loss[-1] > last_loss[0] + config['plateau_overhead'] or 0.6 > last_loss[-1] > last_loss[0]) and
                    len(last_loss) == config['plateau_length']):
                if lr > config['lr_min']:
                    lr = max(lr / config['lr_decay'], config['lr_min'])
                last_loss = []

        return False, origin_image, config['num_iters']+1, config['num_iters']+1

