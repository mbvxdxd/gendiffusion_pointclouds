import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae_pointnet import PointNetfeat, feature_transform_reguliarzer, STNkd
from .encoders import *
from .diffusion import *

class PointNetCls(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, kl_weight, writer=None, it=None):
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)

        log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (- log_pz - entropy).mean()

        logits = self.diffusion(z)

        labels = self.args.labels
        loss_classification = F.cross_entropy(logits, y)