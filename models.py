import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchFormer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return self.norm(attn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# CVAE 编码器
class CVAE_Encoder(nn.Module):
    def __init__(self, in_ch=1, cond_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+cond_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128*16*16, 256)  # 假设输入为64x64
        self.fc_logvar = nn.Linear(128*16*16, 256)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

# CVAE 解码器
class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim=256, cond_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim+cond_ch, 128*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z, cond):
        cond = cond.view(cond.size(0), -1)
        z = torch.cat([z, cond], dim=1)
        x = self.fc(z)
        x = x.view(-1, 128, 16, 16)
        return self.deconv(x)

# CGAN 生成器
class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim=256, cond_ch=1):
        super().__init__()
        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_dim+cond_ch, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BatchFormer(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, cond):
        cond = cond.unsqueeze(2).unsqueeze(3)
        z = torch.cat([z.unsqueeze(2).unsqueeze(3), cond], dim=1)
        x = self.init(z)
        return self.blocks(x)

# CGAN 判别器
class CGAN_Discriminator(nn.Module):
    def __init__(self, in_ch=1, cond_ch=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch+cond_ch, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)