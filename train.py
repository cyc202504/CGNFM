import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# train.py
def train_epoch(cvae_enc, cvae_dec, generator, discriminator,
                dataloader, cvae_opt, g_opt, d_opt, device):
    cvae_enc.train()
    cvae_dec.train()
    generator.train()
    discriminator.train()

    for batch in dataloader:
        # 数据准备
        prev_fine = batch['prev_fine'].to(device)
        next_fine = batch['next_fine'].to(device)
        coarse = batch['coarse'].to(device)
        target = batch['target'].to(device)
        batch_size = target.size(0)

        # ========== 训练CVAE ==========
        # 拼接时间序列数据
        temporal_input = torch.cat([prev_fine, next_fine], dim=1)

        # 编码器前向
        mu, logvar = cvae_enc(temporal_input, coarse)
        z = reparameterize(mu, logvar)

        # 解码器重建
        recon = cvae_dec(z, coarse)

        # 计算复合损失
        huber_loss = F.huber_loss(recon, target)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        cvae_loss = huber_loss + kl_loss

        # 反向传播
        cvae_opt.zero_grad()
        cvae_loss.backward()
        cvae_opt.step()

        # ========== 训练CGAN ==========
        # 准备标签数据
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # 训练判别器
        with torch.no_grad():
            fake_hr = generator(z, coarse)

        # 真实数据判别
        real_pred = discriminator(target, coarse)
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels)

        # 生成数据判别
        fake_pred = discriminator(fake_hr.detach(), coarse)
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels)

        # 合并判别器损失
        d_loss = d_loss_real + d_loss_fake

        # 判别器反向传播
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # 训练生成器
        fake_pred = discriminator(fake_hr, coarse)
        g_loss = F.binary_cross_entropy(fake_pred, real_labels)

        # 生成器反向传播
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std