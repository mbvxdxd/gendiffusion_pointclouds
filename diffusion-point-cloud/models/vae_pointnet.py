import torch
from torch.nn import Module
from .common import *
from .encoders import *
from .diffusion import *
from .pointnet import *

class PointNetVAE(Module):

    def __init__(self, args):
        super().__init__()
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
        self.pointnet_classifier = PointNet(k=args.k, feature_transform=args.feature_transform)

    def get_loss(self, x, y, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            y:  Target class labels, (B, ).
        """
        # Forward pass through encoder
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)

        # Reconstruction loss
        recon_loss = self.diffusion.get_loss(x, z)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_sigma - z_mu.pow(2) - z_sigma.exp()) / x.size(0)
        
        # Forward pass through classifier
        pred, _, _ = self.pointnet_classifier(x)
        # Classification loss
        # target should be of shape (B, 2)
        # since x is of shape (B, N, d), target should be of shape (B, N)
        target = y
        #print(target)
        #print(pred)
        classification_loss = F.cross_entropy(pred, target)
        
        # Combined loss
        loss = recon_loss + kl_weight * kl_loss + classification_loss
        
        # Log the losses if writer is provided
        if writer is not None and it is not None:
            writer.add_scalar('Loss/recon_loss', recon_loss.item(), it)
            writer.add_scalar('Loss/kl_loss', kl_loss.item(), it)
            writer.add_scalar('Loss/classification_loss', classification_loss.item(), it)
            writer.add_scalar('Loss/total_loss', loss.item(), it)
        
        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None, target_class=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
            target_class: Desired class label for guidance.
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)

        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)

        if target_class is not None:
            samples = self.guided_sampling(samples, target_class)

        return samples

    def guided_sampling(self, samples, target_class, num_guidance_steps=5, guidance_scale=1):
        """
        Args:
            samples: Initial samples from the diffusion process, (B, N, d).
            target_class: Desired class label for guidance.
            num_guidance_steps: Number of guidance steps to apply.
            guidance_scale: Scale factor for the guidance adjustments.
        """
        samples.requires_grad_(True)

        for _ in range(num_guidance_steps):
            logits, _, _ = self.pointnet_classifier(samples)
            loss = torch.nn.CrossEntropyLoss()(logits, target_class)
            loss.backward()

            with torch.no_grad():
                samples = samples - guidance_scale * samples.grad
                samples.grad.zero_()

        return samples.detach()
