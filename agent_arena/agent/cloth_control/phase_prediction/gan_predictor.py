import torch.nn as nn
import torch
import numpy as np

from utilities.networks.register import Name2Network
from utilities.networks.image_decoder import ImageDecoder
from utilities.networks.utils import *
from utilities.text import bold


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Name2Network[config.image_encoder.name](**config.image_encoder.params)
        self.classifier_stop_grad = config.classifier.stop_grad
        #print('classifier stop grad', self.classifier_stop_grad)
        self.classifier = nn.Sequential(
            build_mlp(config.classifier.layers, config.classifier.activation_function),
            ### SoftMax Layer for classitransporter.utilsfication
            nn.Softmax(dim=1)
        )
        self.config = config
        if 'image_decoder' in self.config:
            #print('HELLLOOOO')
            self.decoder = ImageDecoder(**config.image_decoder)
            
    
    def forward(self, x):
        #print('x shape', x.shape)
        z = self.encoder(x).view(x.size(0), -1)
        #print('z shape', z.shape)
        z_d = ACTIVATIONS[self.config.act_before_decoder]()(z)

        if self.classifier_stop_grad:
            z_c = z.detach().clone()
        else:
            z_c = z
        
        z_c = ACTIVATIONS[self.config.act_before_classifier]()(z_c)
        y = self.classifier(z_c)
        
        if 'image_decoder' in self.config:
            x_hat = self.decoder(z_d)
            #print('x_hat shape', x_hat.shape)

            # plot the first sample of the batch before and after reconstruction
            # from matplotlib import pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.imshow(((x[0].permute(1, 2, 0).detach().cpu().numpy() + 0.5)*255).astype(np.uint8))
            # plt.subplot(1, 2, 2)
            # plt.imshow(((x_hat[0].permute(1, 2, 0).detach().cpu().numpy() + 0.5)*255).astype(np.uint8))
            # plt.show()

            return {
                'phase_pred': y,
                'recon': x_hat
            }
        return {
            'phase_pred': y
        }

class GAN_Predictor():
    def __init__(self, config):
        self.model = Network(config)
        self.model.to(config.device)
    
        self.config = config
        if 'weight_balancing' in config:
            self.class_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(config.class_weights).to(config.device),
                reduction="mean")
        else:
            self.class_loss = nn.CrossEntropyLoss(reduction="mean")

        if self.config.recon_loss == 'bce':
            pos_weight = torch.tensor(config.class_weights).to(config.device)
            ## pos_weight has the shap C, and I want make it C*H*W using config.out_shape
            pos_weight = pos_weight.view(-1, 1, 1).expand(-1, config.out_shape[0], config.out_shape[1])
            self.recon_loss = nn.BCEWithLogitsLoss(reduction="mean", 
                pos_weight=pos_weight)
        else:
            self.recon_loss = lambda x, y: F.mse_loss(
                x, y,
                reduction='none').sum(dim=(1, 2, 3)).mean()

        #self.metric = MeanMetrics()
        self.optimiser = OPTIMISERS[config.optimiser.name](
            self.model.parameters(), 
            **config.optimiser.params)
        self.param_list = list(self.model.parameters())
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device

    def forward(self, x):
        return self.model(x)
    
    def reconstruct(self, x):
        self.eval_mode()
        with torch.no_grad():
            return self.forward(x)['recon'].detach().cpu().numpy()
    
    def train_block(self, obs, phase, phase_weights=None):
        lossses = {}
        output = self.forward(obs['input'])
        # print('output shape', output['phase_pred'].shape)
        # print('phase shape', phase.shape)
        
        lossses['class_loss'] = self.class_loss(output['phase_pred'], phase)
       
        #lossses['total_loss'] = self.config.class_factor* lossses['class_loss']
        
        if 'recon' in output.keys():
            lossses['recon_loss'] = self.recon_loss(output['recon'], obs['output'])
            #lossses['total_loss'] += 1.0* lossses['recon_loss']
        
        return lossses

    def train(self, obs, phase, update_step, total_steps):

        #self.metric.reset()
        self.train_mode()
        obs = {key: np_to_ts(value, self.device) for key, value in obs.items()}
        phase = np_to_ts(phase, self.device)
        
        self.optimiser.zero_grad()

        losses = self.train_block(obs, phase)

        start = self.config.class_start_update_step
        class_factor = self.config.class_factor if update_step > start else 0.
        if self.config.class_factor_warmup:
            class_factor = max((update_step-start), 0)/(total_steps-start) \
                * self.config.class_factor
        
        losses['class_factor'] = torch.tensor(class_factor)
        
        recon_end = self.config.recon_end_update_step
        recon_factor = self.config.recon_factor if update_step < recon_end else 0.
        losses['recon_factor'] = torch.tensor(recon_factor)

        if update_step == recon_end:
            ## freeze encoder and decoder
            print('freeze encoder and decoder')
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            
            self.param_list = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self.optimiser = OPTIMISERS[self.config.optimiser.name](
                self.param_list, 
                **self.config.optimiser.params)
            

        

           
        losses['total_loss'] = class_factor * losses['class_loss']
        
        if update_step < recon_end:
            losses['total_loss'] += recon_factor* losses['recon_loss']
        #recon_factor* losses['recon_loss']

        # ## check if total loss has gradient
        if losses['total_loss'].requires_grad:
            #print('here')
            losses['total_loss'].backward()

            nn.utils.clip_grad_norm_(self.param_list, self.grad_clip_norm, norm_type=2)
            self.optimiser.step()
    
        #self.metric(loss)
        return {loss_name: np.float32(loss.detach().cpu().numpy()) \
                for loss_name, loss in losses.items()}


    def test(self, obs, phase):
        obs = {key: np_to_ts(value, self.device) for key, value in obs.items()}
        phase = np_to_ts(phase, self.device)

        self.eval_mode()
        with torch.no_grad():
            losses = self.train_block(obs, phase)

        return {loss_name: np.float32(loss.detach().cpu().numpy()) \
                for loss_name, loss in losses.items()}

    def predict(self, obs):
        
        self.eval_mode()
        with torch.no_grad():
            output = self.forward(obs)
        
        print('output shape', output['phase_pred'].shape)
        
        return F.softmax(output['phase_pred'], dim=1).detach().cpu().numpy()
    
    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load(self, path, verbose=False):
        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('phase_predictor')} model on {bold(device)} from {bold(path)}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        #self.optimiser.load_state_dict(checkpoint['optimiser'])

    def save(self, filename, verbose=False):
        
        if verbose:
            print(f"Saving phase_redictor model to {bold(filename)}")
        
        torch.save({
            'model': self.model.state_dict(),
            'optimiser': self.optimiser.state_dict()
        }, filename)