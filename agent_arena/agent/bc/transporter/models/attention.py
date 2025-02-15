# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Attention module."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange

from agent_arena.utilities.networks.utils import *

from ..utils import utils, MeanMetrics, to_device
from ..utils.text import bold
from ..utils.utils import apply_rotations_to_tensor
from ..models.resnet import Name2ResNet



# # REMOVE BELOW
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class Attention:
    """Attention module."""

    def __init__(self, in_shape, n_rotations,
                 encoder_version='resnet43', verbose=False, optimiser={
                     'name': 'adam',
                     'params': {'lr': 1e-4}
                 }):
        # print('Attention')
        # print('in_shape', in_shape)
        # print('n_rotations', n_rotations)
        # print('lite', lite)
        # print('verbose', verbose)
        self.n_rotations = n_rotations
        # self.preprocess = preprocess

        max_dim = np.max(in_shape[:2])
        self.in_shape = in_shape

        self.padding = np.zeros((4, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[1:3] = pad.reshape(2, 1)
        #print('lite', lite)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        model_type = Name2ResNet[encoder_version]
        #ResNet36_4s if lite else ResNet43_8s
        self.model = model_type(in_shape[2], 1)

        self.device = to_device([self.model], "Attention", verbose=verbose)

        self.optimizer = OPTIMISERS[optimiser['name']](
            self.model.parameters(), 
            **optimiser['params'])
        
        #optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.metric = MeanMetrics()

    def forward(self, in_img, softmax=True):
        """Forward pass."""
        B, H, W = in_img.shape[:3]
        in_img = in_img.reshape(B, H, W, -1)
        #print('in_image', in_img.shape)
        in_data = np.pad(in_img, self.padding, mode='constant')
        # in_data = self.preprocess(in_data)
        # print('input data shape', in_data.shape)
        # in_shape = (1,) + in_data.shape
        # in_data = in_data.reshape(in_shape)
        in_tens = torch.tensor(in_data, dtype=torch.float32, requires_grad=False).to(self.device)
       

        # Rotate input.
        in_tens = apply_rotations_to_tensor(in_tens, self.n_rotations).reshape(-1, *in_tens.shape[1:])
        #print('shape in_tens', in_tens.shape)

        # Forward pass.
        # in_tens = torch.split(in_tens, 1, dim=0)  # (self.num_rotations)
        # logits = ()
        # for x in in_tens:
        #     logits += (self.model(x),)
        # logits = torch.cat(logits, dim=0)

        logits = self.model(in_tens)

        # Rotate back output.
        logits = apply_rotations_to_tensor(
            logits, self.n_rotations, reverse=True).reshape(-1, *logits.shape[1:])

        c0 = self.padding[1:3, 0]
        c1 = c0 + in_img.shape[1:3]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        output = Rearrange('b h w c -> b (h w c)')(logits)

        if softmax:
            output = nn.Softmax(dim=1)(output)
            output = output.detach().cpu().numpy()
            output = np.float32(output).reshape(logits.shape[1:])
        return output

    def train_block(self, in_img, p, theta):
        in_img = np.stack(in_img)
        p = np.stack(p)
        theta = np.stack(theta)
        B = in_img.shape[0]
    
        output = self.forward(in_img, softmax=False)
        #print('output shape', output.shape)

        # Get label.
        theta_i = theta / (2 * np.pi / self.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.n_rotations
        label_size = (B,) + self.in_shape[:2] + (self.n_rotations,)
        label = np.zeros((B, *self.in_shape[:2], self.n_rotations))

        # TODO: optimise here
        for b in range(B):
            label[b, p[b][0], p[b][1], theta_i[b]] = 1
        label = torch.tensor(label, dtype=torch.float32).to(self.device)

        # print('attention label', label.shape)
        #print('attention output shape', output.shape)


        # Get loss.
        label = Rearrange('b h w c -> b (h w c)')(label)
        label = torch.argmax(label, dim=1)
        #print('label shape', label.shape)

        loss = self.loss(output, label)

        return loss

    # list of images, picks and thetas
    def train(self, in_img, p, theta):
        """Train."""
        self.metric.reset()
        self.train_mode()
        self.optimizer.zero_grad()

      

        loss = self.train_block(in_img, p, theta)
        loss.backward()
        self.optimizer.step()
    
        self.metric(loss)

        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, theta):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, p, theta)

        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load(self, path, verbose=False):
        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('attention')} model on {bold(device)} from {bold(path)}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, filename, verbose=False):
        if verbose:
            print(f"Saving attention model to {bold(filename)}")
        torch.save(self.model.state_dict(), filename)