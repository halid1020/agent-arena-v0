import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from utilities.networks.register import Name2Network
from utilities.networks.utils import *
from utilities.text import bold


class PretrainedEfficientNetPredictor():
    def __init__(self, config):
        self.config = config

        # Load a pre-trained EfficientNet model
        self.model = models.efficientnet_b0(pretrained=True)

        # Modify the classifier
        num_features = self.model.classifier[1].in_features
        layers = [num_features] + config.classifier.layers
        layers.append(config.num_classes)
        self.model.classifier = build_mlp(
            layers, 
            config.classifier.activation,
            config.classifier.dropout)
            ### SoftMax Layer for classitransporter.utilsfication

        self.model.to(config.device)
            
        self.class_loss = nn.CrossEntropyLoss(reduction="mean")

        self.optimiser = OPTIMISERS[config.optimiser.name](
            self.model.parameters(), 
            **config.optimiser.params)
        #self.param_list = list(self.model.parameters()
        #self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device

    def forward(self, x):
        return self.model(x)
    
    def train_block(self, obs, phase):
        lossses = {}
        output = self.forward(obs['input'])
        lossses['class_loss'] = self.class_loss(output, phase)
        
        return lossses

    def train(self, obs, phase, update_step, total_steps):

        #self.metric.reset()
        self.train_mode()
        obs = {key: np_to_ts(value, self.device) for key, value in obs.items()}
        phase = np_to_ts(phase, self.device)
        
        self.optimiser.zero_grad()

        losses = self.train_block(obs, phase)

        
        class_factor = self.config.class_factor
        
        losses['class_factor'] = torch.tensor(class_factor)
        
        losses['total_loss'] = class_factor* losses['class_loss']
        
       
        losses['total_loss'].backward()

        #nn.utils.clip_grad_norm_(self.param_list, self.grad_clip_norm, norm_type=2)
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
        return None

    def predict(self, obs):
        
        self.eval_mode()
        with torch.no_grad():
            output = self.forward(obs)
        
        return F.softmax(output, dim=1).detach().cpu().numpy()

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