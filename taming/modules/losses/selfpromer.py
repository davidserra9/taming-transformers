import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQSelfPromer(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, perceptual_weight=1.0, disc_weight=1.0,
                 disc_in_channels=3, disc_num_layers=3, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_factor=1.0, disc_loss="hinge"):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight

        self.L1 = nn.L1Loss()
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        self.disc_factor = disc_factor
        self.disc_conditional = disc_conditional

