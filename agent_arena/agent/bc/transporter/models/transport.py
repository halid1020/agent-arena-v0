# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport module."""


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from agent_arena.utilities.networks.utils import *

from ..models.resnet import ResNet43_8s, ResNet36_4s
from ..utils import utils, MeanMetrics, to_device
from ..utils.text import bold
from ..utils.utils import apply_rotations_to_tensor


class Transport:
    """Transport module."""

    def __init__(self, in_channels, n_rotations, 
                 crop_size, verbose=False, 
                 name="Transport", key_optimiser={
                     'name': 'adam',
                     'params': {'lr': 1e-4}
                 }, query_optimiser={
                     'name': 'adam',
                     'params': {'lr': 1e-4}
                 }, neg_samples=0,
                ):
        """Transport module for placing.

        Args:
          in_shape: shape of input image.
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
          preprocess: function to preprocess input images.
        """
        self.iters = 0
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        #print('crop size', crop_size)
        #self.preprocess = preprocess

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((4, 2), dtype=int)
        self.padding[1:3, :] = self.pad_size
        self.neg_samples = neg_samples

        # Crop before network (default for Transporters in CoRL submission).
        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        # 2 fully convolutional ResNets with 57 layers and 16-stride
        self.model_query = ResNet43_8s(in_channels, self.output_dim)
        self.model_key = ResNet43_8s(in_channels, self.kernel_dim)

        self.device = to_device(
            [self.model_query, self.model_key], name, verbose=verbose)

        self.optimizer_query = OPTIMISERS[query_optimiser['name']](
            self.model_query.parameters(), 
            **query_optimiser['params'])
        
        self.optimizer_key = OPTIMISERS[key_optimiser['name']](
            self.model_key.parameters(), 
            **key_optimiser['params'])
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.metric = MeanMetrics()

        self.softmax = nn.Softmax(dim=1)


    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        in0 = Rearrange('b h w c -> b c h w')(in0)
        in1 = Rearrange('b h w c -> b c h w')(in1)

        # print('in0 shape', in0.shape)
        # print('in1 shape', in1.shape)
        output = F.conv2d(in0, in1)
        # print('output shape', output.shape)

        if softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b c h w',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)
            #output = output[0, ...]
            #output = output.detach().cpu().numpy()
        return output


    # TODO: deal with NB.
    def forward(self, in_img, p, softmax=True):
        """Forward pass."""
        #print('in_img shape', in_img.shape)
        B, H, W = in_img.shape[:3]
        in_img = in_img.reshape(B, H, W, -1)

        input_data = np.pad(in_img, self.padding, mode='constant')
        #input_data = self.preprocess(img_unprocessed.copy())
       
        in_tensor = torch.tensor(
            input_data, dtype=torch.float32
        ).to(self.device)

        # Rotate crop, let
        p_ = p[:, (1, 0)]
        pivot = p_ + self.pad_size

        # print('pivot shape', pivot.shape)
        # print('p 0', p[0])
        # print('p_ 0', p_[0])
        #pivot = list(np.array([p[1], p[0]]) + self.pad_size)

        # Crop before network (default for Transporters in CoRL submission).
        crop = apply_rotations_to_tensor(
            in_tensor, self.n_rotations, center=pivot)
        #print('crop shape after rotate', crop.shape)
        crop_tmp = []
        for b in range(B):
            crop_tmp.append(crop[:, b, p[b][0]:(p[b][0] + self.crop_size), \
                           p[b][1]:(p[b][1] + self.crop_size), :])
        crop = torch.stack(crop_tmp, dim=1)
        #print('crop shape', crop.shape)

        in_tensor = in_tensor.unsqueeze(0)
        in_tensor = in_tensor.repeat(
            (1, 1, 1, 1, 1))
        
        NB = B*self.n_rotations
        
        logits = self.model_query(in_tensor.reshape(B, *in_tensor.shape[2:]))
        kernel_raw = self.model_key(crop.reshape(NB, *crop.shape[2:]))

        # Crop after network (for receptive field, and more elegant).
        # logits, crop = self.model([in_tensor, in_tensor])
        # # crop = tf.identity(kernel_bef_crop)
        # crop = tf.repeat(crop, repeats=self.n_rotations, axis=0)
        # crop = tfa_image.transform(crop, rvecs, interpolation='NEAREST')
        # kernel_raw = crop[:, p[0]:(p[0] + self.crop_size),
        #                   p[1]:(p[1] + self.crop_size), :]

        # Obtain kernels for cross-convolution.
        # Padding of one on right and bottom of (h, w)
        kernel_paddings = nn.ConstantPad2d((0, 0, 0, 1, 0, 1, 0, 0), 0)
        kernel = kernel_paddings(kernel_raw)

        logits = logits.reshape(1, B, *logits.shape[1:])
        kernel = kernel.reshape(self.n_rotations, B, *kernel.shape[1:])

        # print('logits shape', logits.shape)
        # print('B', B)
        correlate = []
        for b in range(B):
            correlate.append(self.correlate(logits[:, b, ...], kernel[:, b, ...], softmax))

        #print('correlate shape', correlate[0].shape)
        correlate = torch.stack(correlate)
        
        correlate = correlate.reshape(B, *correlate.shape[2:])
        
        #print('correlate shape', correlate.shape)
        return correlate

    def train_block(self, in_img, p, q, theta):
        in_img = np.stack(in_img)
        p = np.stack(p)
        q = np.stack(q)
        theta = np.stack(theta)
        B, H, W = in_img.shape[:3]
    
        pos_output = self.forward(in_img, p, softmax=False)
        # print('transport output shape', output.shape)
        pos_output = Rearrange('b theta h w -> b (h w theta)')(pos_output)

        #output = torch.tensor(output, dtype=torch.float32).to(self.device)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Get one-hot pixel label map for poxistive picks
        pos_label = np.zeros((*in_img.shape[:3], self.n_rotations))
        for b in range(B):
            pos_label[b, q[b][0], q[b][1], itheta[b]] = 1
        pos_label = torch.tensor(pos_label, dtype=torch.float32).to(self.device)
        #print('pos label shape', pos_label.shape)
        pos_label = Rearrange('b h w c -> b (h w c)')(pos_label)
        pos_label = torch.argmax(pos_label, dim=1)
        pos_loss = self.loss(pos_output, pos_label)

        # Get unifrom pixel label map for negative picks
        if self.neg_samples != 0:
            NB = B*self.neg_samples
            neg_p = np.random.randint(low=0, high=[H, W], size=(NB, 2)).reshape(B, self.neg_samples, 2)
            mask = np.all(neg_p == p[:, np.newaxis, :], axis=2)
            while np.any(mask):
                neg_p[mask] = np.random.randint(low=0, high=[H, W], size=(np.sum(mask), 2))
                mask = np.all(neg_p == p[:, np.newaxis, :], axis=2)
            neg_p = neg_p.reshape(NB, 2)

            # make input image as B x 1 x H x W x C and neg_input image as B x neg_samples x H x W x C
            neg_input_image = np.tile(in_img[:, np.newaxis, :, :, :], (1, self.neg_samples, 1, 1, 1)).reshape(NB, H, W, -1)
            neg_output = self.forward(neg_input_image, neg_p, softmax=True)
            neg_output = Rearrange('b theta h w -> b (h w theta)')(neg_output)

            neg_label = np.ones((NB, H, W, self.n_rotations))/(1.0*H*W*self.n_rotations)
            neg_label = torch.tensor(neg_label, dtype=torch.float32).to(self.device)
            neg_label = Rearrange('b h w c -> b (h w c)')(neg_label)

            neg_loss = self.loss(neg_output, neg_label)

            return pos_loss + neg_loss

        else:
            return pos_loss
        
    def train(self, in_img, p, q, theta):
        """Transport pixel p to pixel q.

        Args:
          in_img: input image.
          p: pixel (y, x)
          q: pixel (y, x)
          theta: rotation label in radians.
          backprop: True if backpropagating gradients.

        Returns:
          loss: training loss.
        """

        self.metric.reset()
        self.train_mode()
        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()

        loss = self.train_block(in_img, p, q, theta)
        loss.backward()
        self.optimizer_query.step()
        self.optimizer_key.step()
        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, q, theta):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, p, q, theta)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model_query.train()
        self.model_key.train()

    def eval_mode(self):
        self.model_query.eval()
        self.model_key.eval()

    def format_fname(self, fname, is_query):
        suffix = 'query' if is_query else 'key'
        return fname.split('.pth')[0] + f'_{suffix}.pth'

    def load(self, fname, verbose):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('transport query')} model on {bold(device)} from {bold(query_name)}")
            print(
                f"Loading {bold('transport key')}   model on {bold(device)} from {bold(key_name)}")

        self.model_query.load_state_dict(
            torch.load(query_name, map_location=self.device))
        self.model_key.load_state_dict(
            torch.load(key_name, map_location=self.device))

    def save(self, fname, verbose=False):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            print(
                f"Saving {bold('transport query')} model to {bold(query_name)}")
            print(
                f"Saving {bold('transport key')}   model to {bold(key_name)}")

        torch.save(self.model_query.state_dict(), query_name)
        torch.save(self.model_key.state_dict(), key_name)