import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

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
    def __init__(self, disc_start, codebook_weight=1.0, ssim_weight=1.0, perceptual_weight=1.0, disc_weight=1.0,
                 disc_in_channels=3, disc_num_layers=3, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_factor=1.0, disc_loss="hinge"):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.disc_weight = disc_weight

        self.perceptual_loss = LPIPS().eval()
        self.ssim = StructuralSimilarityIndexMeasure(kernel_size=11).cuda()
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

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, targets, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        l1_loss = torch.mean((targets.contiguous() - reconstructions.contiguous()) ** 2)
        ssim_loss = 1 - self.ssim(targets.contiguous() - reconstructions.contiguous())
        perceptual_loss = torch.mean(self.perceptual_loss(targets.contiguous(), reconstructions.contiguous()))

        nll_loss = l1_loss + self.ssim_weight * ssim_loss + self.perceptual_weight * perceptual_loss

        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
        return loss, log