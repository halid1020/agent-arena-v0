import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_arena.agent.utilities.torch_utils import *

class ImageDecoder(nn.Module):
        __constants__ = ['embedding_size', 'image_dim']

        def __init__(self, belief_size, state_size, embedding_size, image_dim, activation_function='relu', batchnorm=False, output_mode=None):
            super().__init__()
            self.image_dim = image_dim
            self.act_fn = getattr(F, activation_function)
            self.embedding_size = embedding_size
            self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
            self.output_mode = output_mode
            if image_dim[1] == 128:
                self._decoder = nn.Sequential(
                    nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                    nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(128, 64, 5, stride=2),
                    nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(64, 32, 5, stride=2),
                    nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(32, 16, 6, stride=2),
                    nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(16, self.image_dim[0], 6, stride=2)
                )
            elif image_dim[1] == 64:
                self._decoder = nn.Sequential(
                    nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                    nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(128, 64, 5, stride=2),
                    nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(64, 32, 6, stride=2),
                    nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                    ACTIVATIONS[activation_function](),
                    nn.ConvTranspose2d(32, self.image_dim[0], 6, stride=2)
                )


        def forward(self, belief, latent_state):
            x = torch.cat([belief, latent_state], dim=-1)
            batch_shape = x.shape[:-1]
            embed_size = x.shape[-1]
            squeezed_size = np.prod(batch_shape).item()
            x = x.reshape(squeezed_size, embed_size)

            hidden = self.fc1(x)  # No nonlinearity here
            hidden = hidden.view(-1, self.embedding_size, 1, 1)
            x = self._decoder(hidden)

            shape = x.shape[1:]
            x = x.reshape((*batch_shape, *shape))

            if self.output_mode == 'normal':
                x = td.Independent(td.Normal(x, 1), len(shape))

            return x



class ImageEncoder(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, activation_function='relu', batchnorm=False, residual=False):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size

        if self.image_dim[1] == 128:
            self._encoder = nn.Sequential(
                nn.Conv2d(self.image_dim[0], 16, 4, stride=2) if not residual else ResidualBlock(self.image_dim[0], 16, kernel=4, stride=2),
                nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(16, 32, 4, stride=2)  if not residual else ResidualBlock(16, 32, kernel=4, stride=2),
                nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(32, 64, 4, stride=2)  if not residual else ResidualBlock(32, 64, kernel=4, stride=2),
                nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(64, 128, 4, stride=2) if not residual else ResidualBlock(64, 128, kernel=4, stride=2),
                nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(128, 256, 4, stride=2) if not residual else ResidualBlock(128, 256, kernel=4, stride=2),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function]()
            )
        elif self.image_dim[1] == 64:
            self._encoder = nn.Sequential(
                nn.Conv2d(self.image_dim[0], 32, 4, stride=2) if not residual else ResidualBlock(self.image_dim[0], 32, kernel=4, stride=2),
                nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(32, 64, 4, stride=2) if not residual else ResidualBlock(32, 64, kernel=4, stride=2),
                nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(64, 128, 4, stride=2) if not residual else ResidualBlock(64, 128, kernel=4, stride=2),
                nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function](),
                nn.Conv2d(128, 256, 4, stride=2) if not residual else ResidualBlock(128, 256, kernel=4, stride=2),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                ACTIVATIONS[activation_function]()
            )
        else:
            raise NotImplementedError
        
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        #print('x shape', x.shape)
        batch_shape = x.shape[:-3]
        embed_size = x.shape[-3:]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape((squeezed_size, *embed_size))
        hidden = self._encoder(x)
        hidden = hidden.reshape(-1, 1024)
        output = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        shape = output.shape[1:]

        # print('batch_shape', batch_shape)
        # print('shape', shape)

        output = output.reshape((*batch_shape, *shape))

        return output
    

class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, kernel=3, stride=1, padding=0): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel, 
                                         stride=stride, padding=padding, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), # can optionally do the operation in-place. Default: False ???
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: # When we cannot do F(x) + x directly, using following to cheat.
            
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 
                                                 kernel_size=kernel, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
    
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out