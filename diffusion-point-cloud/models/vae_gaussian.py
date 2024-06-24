import torch
from torch.nn import Module
import numpy as np

from .common import *
from .encoders import *
from .diffusion import *
from .pointnet import *


class GaussianVAE(Module):

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
        
    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

class GuidedGaussianVAE(GaussianVAE):

    def __init__(self, args, alpha=0.5, size_average=True):
        super().__init__(args)
        self.alpha = alpha
        self.gamma = 2
        self.size_average = size_average
        self.reg_weight = args.kl_weight
        # Assuming alpha is meant for class weighting and is a list or tensor of shape [C]
        if isinstance(alpha, (list, np.ndarray, torch.Tensor)):
            class_weights = torch.tensor(alpha, dtype=torch.float)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def apply_guidance(self, x_diffused, classification):
        if isinstance(classification, tuple):
            logits = classification[0]
        else:
            logits = classification
        
        confidence, _ = logits.max(dim=1, keepdim=True)
        guided_x = (1 - self.guidance_weight) * x_diffused + self.guidance_weight * x_diffused * confidence.unsqueeze(-1)
        
        return guided_x

    def get_loss(self, x, labels, A, writer=None, it=None):
        bs = x.size()[0]
        num_classes = 2  # Assuming binary classification as indicated by the expected target size [128, 2]

        # Check if labels are already one-hot encoded
        if labels.dim() == 1 or labels.size(1) != num_classes:
            # Convert scalar labels to one-hot encoded format
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        else:
            labels_one_hot = labels
        # get Balanced Cross Entropy Loss
        labels = labels.long()
        ce_loss = self.cross_entropy_loss(x, labels_one_hot)
        
        x = x.contiguous().view(-1, num_classes)
        # get predicted class probabilities for the true class
        pn = F.softmax(x)
        pn = pn.gather(1, labels.view(-1, 1)).view(-1)

        # get regularization term
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1) # .to(device)
            if A.is_cuda: I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight*reg/bs
        else:
            reg = 0

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)

        if writer is not None:
            writer.add_scalar('train/loss', loss.mean(), it)
            writer.add_scalar('train/ce_loss', ce_loss.mean(), it)
            writer.add_scalar('train/reg_loss', reg, it)

        if self.size_average: return loss.mean() + reg
        else: return loss.sum() + reg

    def sample(self, z, num_points, flexibility, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples




