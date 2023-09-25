import torch.nn as nn
import torch
from utils.encoder_utils import Reshape, Layer64, Layer64Transpose
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_sig_dim(k, p, n, input_dim):
    sig_dim = np.ceil((input_dim - 3 * (k - 1) + 2 * p) / 2)
    for _ in range(n + 1):
        sig_dim -= k - 1 - 2 * p
        sig_dim = sig_dim / 2
        sig_dim = np.ceil(sig_dim)

    return int(sig_dim)


class VAE_2D(nn.Module):
    def __init__(self, k, n, latent_dim, input_dim, lvef=False, c=12, pred2=2):
        super().__init__()

        self.p = int((k - 1) / 2)
        self.c = c
        self.pred2 = pred2
        self.sig_dim = calc_sig_dim(k, self.p, n, input_dim)
        self.flatten_dim = 64 * self.sig_dim * c
        l = n + 2
        ls = sum([2**i for i in range(l)])
        self.final_dim = (
            (2**l) * self.sig_dim + (2 + ls) * (k - 2) + 2 - (2 * ls * self.p)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, stride=(1, 1), kernel_size=(1, k), padding=(0, 0)),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.01),
            nn.Conv2d(8, 16, stride=(1, 1),
                      kernel_size=(1, k), padding=(0, 0)),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, stride=(1, 2), kernel_size=(
                1, k), padding=(0, self.p)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(1, 2), kernel_size=(
                1, k), padding=(0, self.p)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01),
            Layer64(k, self.p, n, conv2d=True),
            nn.Flatten(),
        )

        self.z_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.z_log_var = nn.Linear(self.flatten_dim, latent_dim)
        self.regressor = nn.Linear(self.pred2 + 2, 3)
        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            Reshape(-1, 64, c, self.sig_dim),
            nn.InstanceNorm2d(64),
            Layer64Transpose(k, self.p, n, conv2d=True),
            nn.ConvTranspose2d(
                64, 32, stride=(1, 2), kernel_size=(1, k), padding=(0, self.p)
            ),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                32, 16, stride=(1, 2), kernel_size=(1, k), padding=(0, self.p)
            ),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                16, 8, stride=(1, 1), kernel_size=(1, k), padding=(0, 0)
            ),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(8, 1, stride=(
                1, 1), kernel_size=(1, k), padding=(0, 0)),
            nn.Linear(self.final_dim, input_dim),
        )

    def encoding_fn(self, x):
        x = x[:, :, :self.c, :400]
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z

    def forward(self, x):
        rri = x[:, :, [0, 6], 400]
        x = x[:, :, :self.c, :400]

        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        if rri.shape[0] == 1:
            rri = rri.squeeze().reshape(1, 2)
        else:
            rri = rri.squeeze()
        prediction_input = torch.cat((encoded[:, :self.pred2], rri), 1)
        prediction = self.regressor(prediction_input)
        pred_inf_input = torch.cat((z_mean[:, :self.pred2], rri), 1)
          
        prediction = self.sigmoid(prediction)
        return encoded, z_mean, z_log_var, decoded, prediction, pred_inf_input

    def inference_pred(self, x):
        rri = x[:, :, [0, 6], 400]
        x = x[:, :, :self.c, :400]

        x = self.encoder(x)
        z_mean = self.z_mean(x)
        if rri.shape[0] == 1:
            rri = rri.squeeze().reshape(1, 2)
        else:
            rri = rri.squeeze()
        prediction_input = torch.cat((z_mean[:, :self.pred2], rri), 1)
        prediction = self.regressor(prediction_input)
        return prediction

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded