import torch.nn as nn
import torch
import numpy as np

from utilities.networks.register import Name2Network
from utilities.networks.utils import *
from utilities.text import bold


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.autoencoder = Name2Network['unet'](**config.unet.params)
        self.classifier_stop_grad = config.classifier.stop_grad
        #print('classifier stop grad', self.classifier_stop_grad)
        self.classifier = nn.Sequential(
            build_mlp(config.classifier.layers, config.classifier.activation_function),
            ### SoftMax Layer for classitransporter.utilsfication
            #nn.Softmax(dim=1)
        )
        self.config = config
        self.flatten = nn.Flatten()
            
    
    def forward(self, x):
        x_hat, latent = self.autoencoder(x)

        z1, z2, z3, z4, z5 = latent
        z4_flat = self.flatten(F.adaptive_avg_pool2d(z4, (1, 1)))
        z5_flat = self.flatten(z5)
        z = torch.cat([z4_flat, z5_flat], dim=1)
        #print('z shape', z.shape)
        if self.classifier_stop_grad:
            z_c = z.detach().clone()
        else:
            z_c = z
        
        z_c = ACTIVATIONS[self.config.act_before_classifier]()(z_c)
        y = self.classifier(z_c)

        return {
            'phase_pred': y,
            'recon': x_hat
        }

class UNet_Predictor():
    def __init__(self, config):
        self.model = Network(config)
        self.model.to(config.device)
    
        self.config = config
        self.class_loss = nn.CrossEntropyLoss(reduction="mean")
        #self.recon_loss = nn.MSELoss(reduction="mean")
        #self.metric = MeanMetrics()
        self.optimiser = OPTIMISERS[config.optimiser.name](
            self.model.parameters(), 
            **config.optimiser.params)
        self.param_list = list(self.model.parameters())
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        if self.config.recon_loss == 'bce':
            pos_weight = torch.tensor(config.class_weights).to(config.device)
            ## pos_weight has the shap C, and I want make it C*H*W using config.out_shape
            pos_weight = pos_weight.view(-1, 1, 1).expand(-1, config.out_shape[0], config.out_shape[1])
            self.recon_loss = nn.BCEWithLogitsLoss(reduction="mean", 
                pos_weight=pos_weight)
        elif self.config.recon_loss == 'mse':
            self.recon_loss = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.model(x)
    
    def train_block(self, obs, phase):
        lossses = {}
        output = self.forward(obs['input'])
        # print('output shape', output['phase_pred'].shape)
        #print('phase shape', phase.shape)
        #print('pred phase', output['phase_pred'].shape)
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
            print('freeze unet')
            for param in self.model.autoencoder.parameters():
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
    
    def reconstruct(self, obs):
        self.eval_mode()
        with torch.no_grad():
            output = self.forward(obs)
        
        if  self.config.recon_loss == 'bce':
            return nn.Sigmoid()(output['recon']).detach().cpu().numpy()
        
        return output['recon'].detach().cpu().numpy()

    def predict(self, obs):
        
        self.eval_mode()
        with torch.no_grad():
            output = self.forward(obs)
        
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