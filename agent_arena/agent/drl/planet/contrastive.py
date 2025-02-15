import torch
import torch.nn as nn

from .networks import ImageEncoder
# from registration.data_transformer import DATA_TRANSFORMER
from agent_arena.agent.utilities.torch_utils import soft_update_params

class ContrastiveEncoder(nn.Module):
    """
    Contrastive encoder for learning representation in PlaNet
    """

    def __init__(self, config):
        super(ContrastiveEncoder, self).__init__()

        self.encoder = ImageEncoder(
            image_dim=config.input_obs_dim,
            embedding_size=config.embedding_dim,
            activation_function=config.activation,
            batchnorm=config.encoder_batchnorm,
            residual=config.encoder_residual
        )

        self.encoder_target = ImageEncoder(
            image_dim=config.input_obs_dim,
            embedding_size=config.embedding_dim,
            activation_function=config.activation,
            batchnorm=config.encoder_batchnorm,
            residual=config.encoder_residual
        )

        # Copy weights from encoder to encoder_target
        for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            target_param.data.copy_(param.data)

        self.projector = nn.Sequential(
            nn.Linear(config.embedding_dim, config.contrastive_dim),
        )

        self.W = nn.Parameter(torch.rand(config.contrastive_dim, config.contrastive_dim))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.anchor_transformer = DATA_TRANSFORMER[config.anchor_transformer.name](
            config.anchor_transformer.params)
        
        self.positive_transformer = DATA_TRANSFORMER[config.positive_transformer.name](
            config.positive_transformer.params)
        
        self.device = config.device
        self.encoder_tau = config.encoder_tau
        

    def forward(self, x):
        return self.encoder(x)

    def project(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                e_out = self.encoder_target(x)
        else:
            e_out = self.encoder(x)
        
        p_out = self.projector(e_out)

        if detach:
            p_out = p_out.detach()
        return p_out

    def compute_logits(self, p_a, p_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix p_a (W p_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, p_pos.T)  # (z_dim,B)
        logits = torch.matmul(p_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    
    def compute_loss(self, obs_anchor, obs_pos):
        
        p_a = self.project(obs_anchor)
        p_pos = self.project(obs_pos, ema=True)

        logits = self.compute_logits(p_a, p_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        return self.cross_entropy_loss(logits, labels)
    
    def sample_pairs(self, images):
        """
        Sample positive and negative pairs
        """
        anchors = self.anchor_transformer(images)
        positives = self.positive_transformer(images)

        return anchors, positives
    
    def update_target(self):
        soft_update_params(
            self.encoder, self.encoder_target,
            self.encoder_tau
        )