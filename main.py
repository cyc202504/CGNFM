# main.py
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data import LSTDataset
from models import CVAE_Encoder, CVAE_Decoder, CGAN_Generator, CGAN_Discriminator
from train import train_epoch
import numpy as np
import random


class Experiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        # 初始化模型
        self.cvae_enc = CVAE_Encoder(in_ch=2 * opt.n_refs, cond_ch=1).to(self.device)  # 输入通道数根据参考影像数量调整
        self.cvae_dec = CVAE_Decoder(latent_dim=256, cond_ch=1).to(self.device)
        self.generator = CGAN_Generator(latent_dim=256, cond_ch=1).to(self.device)
        self.discriminator = CGAN_Discriminator(in_ch=1, cond_ch=1).to(self.device)

        # 多GPU支持
        if opt.ngpu > 1:
            self.cvae_enc = torch.nn.DataParallel(self.cvae_enc)
            self.cvae_dec = torch.nn.DataParallel(self.cvae_dec)
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

        # 优化器
        self.cvae_opt = torch.optim.Adam(
            list(self.cvae_enc.parameters()) + list(self.cvae_dec.parameters()),
            lr=opt.lr
        )
        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=opt.lr)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr)

        # 创建保存目录
        opt.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataloader(self, data_dir, mode='train'):
        """统一数据加载方法"""
        dataset = LSTDataset(
            root_dir=data_dir,
            image_size=self.opt.image_size,
            patch_size=self.opt.patch_size,
            n_refs=self.opt.n_refs,
            stride=self.opt.patch_stride if mode == 'train' else self.opt.patch_size
        )
        return DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=(mode == 'train'),
            num_workers=self.opt.num_workers,
            pin_memory=True
        )

    def train(self, train_dir, val_dir, epochs):
        # 初始化数据集
        train_loader = self._get_dataloader(train_dir, 'train')
        val_loader = self._get_dataloader(val_dir, 'val')

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self._train_epoch(train_loader, epoch)

            # 验证阶段
            val_loss = self._validate(val_loader)
            print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f}")

            # 保存模型
            self._save_checkpoint(epoch)

    def _train_epoch(self, loader, epoch):
        self.cvae_enc.train()
        self.cvae_dec.train()
        self.generator.train()
        self.discriminator.train()

        for batch_idx, data_dict in enumerate(loader):
            # 数据预处理
            ref_coarse = data_dict['ref_0_coarse'].to(self.device)
            ref_fine = data_dict['ref_0_fine'].to(self.device)
            pred_coarse = data_dict['pred_coarse'].to(self.device)
            target = data_dict['pred_fine'].to(self.device)

            # 拼接时间序列输入 (假设使用两个参考影像)
            temporal_input = torch.cat([ref_coarse, ref_fine], dim=1)  # [B, 2*C, H, W]

            # 训练CVAE
            mu, logvar = self.cvae_enc(temporal_input, pred_coarse)
            z = self._reparameterize(mu, logvar)
            recon = self.cvae_dec(z, pred_coarse)

            # 计算CVAE损失
            huber_loss = torch.nn.HuberLoss()(recon, target)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            cvae_loss = huber_loss + kl_loss

            # CVAE反向传播
            self.cvae_opt.zero_grad()
            cvae_loss.backward()
            self.cvae_opt.step()

            # 训练CGAN (省略部分代码，详见原train_epoch逻辑)
            # ... CGAN训练逻辑保持不变，注意输入数据适配

    def _validate(self, loader):
        self.cvae_enc.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data_dict in loader:
                ref_coarse = data_dict['ref_0_coarse'].to(self.device)
                ref_fine = data_dict['ref_0_fine'].to(self.device)
                pred_coarse = data_dict['pred_coarse'].to(self.device)
                target = data_dict['pred_fine'].to(self.device)

                temporal_input = torch.cat([ref_coarse, ref_fine], dim=1)
                mu, logvar = self.cvae_enc(temporal_input, pred_coarse)
                z = self._reparameterize(mu, logvar)
                recon = self.cvae_dec(z, pred_coarse)

                loss = torch.nn.HuberLoss()(recon, target)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _reparameterize(self, mu, logvar):
        """重参数化函数移至类内部"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'cvae_enc': self.cvae_enc.state_dict(),
            'cvae_dec': self.cvae_dec.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizers': {
                'cvae': self.cvae_opt.state_dict(),
                'generator': self.g_opt.state_dict(),
                'discriminator': self.d_opt.state_dict()
            }
        }
        torch.save(checkpoint, self.opt.save_dir / f'checkpoint_epoch_{epoch + 1}.pth')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LST时空融合训练参数')
    # 核心参数
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--n_refs', type=int, default=1, choices=[1, 2], help='参考影像数量')

    # 设备参数
    parser.add_argument('--cuda', action='store_true', default=True, help='启用CUDA')
    parser.add_argument('--ngpu', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # 路径参数
    parser.add_argument('--train_dir', type=Path, required=True, help='训练数据路径')
    parser.add_argument('--val_dir', type=Path, required=True, help='验证数据路径')
    parser.add_argument('--test_dir', type=Path, help='测试数据路径')
    parser.add_argument('--save_dir', type=Path, default=Path('outputs'), help='模型保存路径')

    # 数据参数
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='原始影像尺寸 (高度, 宽度)')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[64, 64],
                        help='训练用patch尺寸')
    parser.add_argument('--patch_stride', type=int, nargs=2, default=[64, 64],
                        help='训练用patch步长')

    opt = parser.parse_args()

    # 初始化设置
    set_seed(2021)
    if opt.cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA不可用但请求使用GPU")
        cudnn.benchmark = True
        cudnn.deterministic = True

    # 运行实验
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir, opt.epochs)

    # 可选测试逻辑
    if opt.test_dir:
        experiment.test(opt.test_dir)