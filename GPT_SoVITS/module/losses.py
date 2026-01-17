import math

import torch


Tensor = torch.Tensor


def feature_loss(fmap_r, fmap_g):
    loss = torch.tensor(0).to(fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g, strict=False):
        for rl, gl in zip(dr, dg, strict=False):
            rl = rl.float().detach()
            gl = gl.float()
            loss = torch.mean(torch.abs(rl - gl)) + loss

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = torch.tensor(0).to(disc_real_outputs[0].device)
    r_losses: list[float] = []
    g_losses: list[float] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=False):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = r_loss + g_loss + loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = torch.tensor(0).to(disc_outputs[0].device)
    gen_losses: list[Tensor] = []
    for dg in disc_outputs:
        dg = dg.float()
        l_m = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_m)
        loss = l_m + loss

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl


def mle_loss(z, m, logs, logdet, mask):
    loss = torch.sum(logs) + 0.5 * torch.sum(
        torch.exp(-2 * logs) * ((z - m) ** 2)
    )  # neg normal likelihood w/o the constant term
    loss = loss - torch.sum(logdet)  # log jacobian determinant
    loss = loss / torch.sum(torch.ones_like(z) * mask)  # averaging across batch, channel and time axes
    loss = loss + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return loss
