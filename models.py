import torch.nn as nn
import torch
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal

from math import pi
from scipy.special import logsumexp
from utils import calculate_matmul, calculate_matmul_n_times

# import tensorflow as tf


# models.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class GaussianDensityEstimator(nn.Module):
    """
    A single Gaussian density estimator.
    """
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.fitted = False

    def fit(self, x):
        """
        Fit a single multivariate Gaussian to the input features.
        x: Tensor of shape (n_samples, input_dim)
        """
        with torch.no_grad():
            mu = x.mean(dim=0)
            x_centered = x - mu
            cov = (x_centered.T @ x_centered) / (x.shape[0] - 1)
            cov += self.eps * torch.eye(self.input_dim, device=x.device)

            self.dist = MultivariateNormal(loc=mu, covariance_matrix=cov)
            self.fitted = True

    def score_samples(self, x):
        """
        Compute log-likelihood of samples under the estimated Gaussian.
        x: Tensor of shape (n_samples, input_dim)
        Returns: log_probs of shape (n_samples,)
        """
        if not self.fitted:
            raise RuntimeError("Call .fit(x) before scoring.")
        return self.dist.log_prob(x)


class CNN(nn.Module):
    def __init__(self, num_channels=19, contig_len=250, out_dim=2, conf=[]):
        super(CNN, self).__init__()
        
        conv_layers = []
        in_channels = num_channels
        conv_layers_conf = conf[0]
        for layer in conv_layers_conf:
            out_channels, kernel, relu, bn, mp = layer
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel))
            if relu:
                conv_layers.append(nn.ReLU())
            if bn:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            if mp:
                conv_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        fc_layers_conf = conf[1]
        
        num_in = fc_layers_conf[0]
        
        fc_layers = []
        for layer_size in fc_layers_conf[1:]:
            fc_layers.append(nn.Linear(num_in, layer_size))
            fc_layers.append(nn.ReLU())
            num_in = layer_size
        
        fc_layers.append(nn.Linear(num_in, out_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        
#         in_lin, out_lin = 25, 8
#         if kernel == 16:
#             in_lin, out_lin = 17, 8
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(num_channels, num_channels, kernel),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels, num_channels * 2, kernel // 2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels * 2, num_channels * 4, kernel // 2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(num_channels * 4, 1, kernel // 2),
#             nn.ReLU()
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_lin, out_lin),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU(),
#             nn.Linear(4, out_dim)
#         )

    def forward(self, x):
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     print (x.shape)

        # x = x.view(x.shape[0], -1)
        # for layer in self.fc_layers:
        #     x = layer(x)
        #     print (x.shape)

        # input()

        out = self.conv_layers(x)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out
    

class CouplingLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('mask', mask)
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, invert=False):
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)

        if not invert:
            y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = torch.sum(s, dim=1)
            return y, log_det
        else:
            y = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
            return y


class RealNVP(nn.Module):
    def __init__(self, num_coupling_layers, input_dim, hidden_dim=512):
        super(RealNVP, self).__init__()
        self.num_coupling_layers = num_coupling_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.coupling_layers = nn.ModuleList()

        mask = torch.zeros(input_dim)
        mask[::2] = 1

        for i in range(num_coupling_layers):
            self.coupling_layers.append(CouplingLayer(input_dim, hidden_dim, mask))
            mask = 1 - mask

        self.register_buffer('prior_mean', torch.zeros(input_dim))
        self.register_buffer('prior_std', torch.ones(input_dim))

    def forward(self, x):
        z = x
        log_det = 0

        for layer in self.coupling_layers:
            z, ld = layer(z)
            log_det += ld

        return z, log_det

    def inverse(self, z):
        x = z

        for layer in reversed(self.coupling_layers):
            x = layer(x, invert=True)

        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_prob_z = torch.distributions.Normal(self.prior_mean, self.prior_std).log_prob(z).sum(dim=1)
        return log_prob_z + log_det

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.input_dim,
                        device=self.prior_mean.device,
                        dtype=self.prior_mean.dtype)
        x = self.inverse(z)
        return x

    def score_samples(self, x):
        return self.log_prob(x)

    def log_loss(self, x):
        return -self.log_prob(x).mean()
    

# class Coupling(nn.Module):
#     def __init__(self, input_shape):
#         super(Coupling, self).__init__()
#         output_dim = 16
        
#         self.t_layers = nn.Sequential(
#             nn.Linear(input_shape, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, input_shape)
#         )
        
#         self.s_layers = nn.Sequential(
#             nn.Linear(input_shape, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, input_shape),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         t = self.t_layers(x)
#         s = self.s_layers(x)
#         return s, t

# class RealNVP(nn.Module):
#     def __init__(self, num_coupling_layers, input_dim):
#         super(RealNVP, self).__init__()
#         self.num_coupling_layers = num_coupling_layers
        
#         self.distribution = D.MultivariateNormal(torch.zeros(input_dim).cuda(), torch.eye(input_dim).cuda())
#         self.masks = torch.tensor(np.array(
#             [np.concatenate((np.zeros(input_dim // 2), np.ones(input_dim // 2))),
#              np.concatenate((np.ones(input_dim // 2), np.zeros(input_dim // 2)))] * (num_coupling_layers // 2)
#         )).float().cuda()

#         self.loss_tracker = []
#         self.layers_list = nn.ModuleList([Coupling(input_dim).cuda() for _ in range(num_coupling_layers)])

#     def forward(self, x, dir=-1):
#         log_det_inv = 0
#         direction = dir
#         for i in range(self.num_coupling_layers)[::direction]:
#             x_masked = x * self.masks[i]
#             reversed_mask = 1 - self.masks[i]
#             s, t = self.layers_list[i](x_masked)
#             s = s * reversed_mask
#             t = t * reversed_mask
#             gate = (direction - 1) / 2
#             x = (
#                 reversed_mask
#                 * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
#                 + x_masked
#             )
#             log_det_inv += gate * torch.sum(s, dim=1)

#         return x, log_det_inv

#     def log_loss(self, x):
#         y, logdet = self(x)
#         log_likelihood = self.distribution.log_prob(y) + logdet
#         return -torch.mean(log_likelihood)
    
#     def score_samples(self, x):
#         y, logdet = self(x, dir=1)
#         log_likelihood = self.distribution.log_prob(y) + logdet
#         return log_likelihood

#     def train_step(self, data):
#         self.optimizer.zero_grad()
#         loss = self.log_loss(data)
#         loss.backward()
#         self.optimizer.step()
#         self.loss_tracker.append(loss.item())
#         return {"loss": np.mean(self.loss_tracker)}

#     def test_step(self, data):
#         loss = self.log_loss(data)
#         self.loss_tracker.append(loss.item())
#         return {"loss": np.mean(self.loss_tracker)}