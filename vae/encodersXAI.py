import torch
from torch import nn
import yaml
from ecgxai.network.causalcnn import encoder, decoder


class VAE_XAI(nn.Module):
    def __init__(self, k, l, latent_dim, input_dim, lvef=False, c=12, pred2=False):
        super().__init__()

        with open("/exports/lkeb-hpc/vovandervalk/vae-for-ecg/vae/params.yaml", "r") as stream:
            params = yaml.safe_load(stream)

        params["encoder"]["out_channels"] = latent_dim
        params["decoder"]["k"] = latent_dim

        self.encoder = encoder.CausalCNNVEncoder(**params["encoder"])
        self.decoder = decoder.CausalCNNVDecoder(**params["decoder"])
        self.c = c
        self.pred2 = pred2
        self.regressor = nn.Linear(latent_dim + 2, 3)
        self.regressor2 = nn.Linear(4, 3)
        self.sigmoid = nn.Sigmoid()

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(1, 1, stride=(2, 1), kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01)
        )
        self.post_decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 1, stride=(
                2, 1), kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01)
        )

    def encoding_fn(self, x):
        x = x[:, :self.c, :400]
        x = self.encoder(x)
        z_mean, z_log_var = x[0], x[1]
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z

    def forward(self, x):
        rri = x[:, [0, 6], 400]
        x = x[:, :self.c, :400]
        if self.c == 24:
            x = self.pre_encoder(x)
        x = self.encoder(x)
        z_mean, z_log_var = x[0], x[1]
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        if self.c == 24:
            decoded = self.post_decoder(decoded[0])
        
        if self.pred2:
            prediction_input = torch.cat((encoded[:, :2], rri), 1)
            prediction = self.regressor2(prediction_input)
            pred_inf_input = torch.cat((z_mean[:, :2], rri), 1)
        else:
            prediction_input = torch.cat((encoded, rri), 1)
            prediction = self.regressor(prediction_input)
            pred_inf_input = torch.cat((z_mean, rri), 1)
            
        return encoded, z_mean, z_log_var, decoded[0], prediction, pred_inf_input
    
    def inference_pred(self, x):
        rri = x[:, :, [0, 6], 400]
        x = x[:, :, :self.c, :400]

        x = self.encoder(x)
        z_mean = self.z_mean(x)
        if rri.shape[0] == 1:
            rri = rri.squeeze().reshape(1, 2)
        else:
            rri = rri.squeeze()

        if self.pred2:
            prediction_input = torch.cat((z_mean[:, :2], rri), 1)
            prediction = self.regressor2(prediction_input)
        else:
            prediction_input = torch.cat((z_mean, rri), 1)
            prediction = self.regressor(prediction_input)
        prediction = self.sigmoid(prediction)
        return prediction
    
    def decode(self, x):
        return self.decoder(x)