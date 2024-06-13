import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import *
from .common import *
from .diffusion import *

class PointNetVAE(nn.Module):
    def __init__(self, args):
        super(PointNetVAE, self).__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)
        log_pz = standard_normal_logprob(z).sum(dim=1)
        entropy = gaussian_entropy(logvar=z_sigma)
        loss_prior = (- log_pz - entropy).mean()
        loss_recons = self.diffusion.get_loss(x, z)
        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None):
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

if __name__ == "__main__":
    class Args:
        latent_dim = 256
        residual = True
        num_steps = 1000
        beta_1 = 0.1
        beta_T = 20.0
        sched_mode = 'linear'

    args = Args()
    model = PointNetVAE(args)
    print(model)
