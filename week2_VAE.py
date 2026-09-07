#%%
""" 0. Reproducibility """
import torch 
from utils import set_random_seed

device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
config={
    "dataset": "redwine",
    "seed": 42,
}
set_random_seed(config["seed"])
#%%
""" 1. Load the raw data """
import pandas as pd
import numpy as np
# import torch 

from datasets.raw_data import load_raw_data


data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config)
features = continuous_features + categorical_features
print(data.head())
# %%
""" 2. Preprocess the raw data """
from datasets.preprocess import CustomDataset

config['train'] = True
config['test_size'] = 0.2
config['seed'] = 42

train_dataset = CustomDataset(
    config,
    cont_scalers=None,
    cat_scalers=None,
    train=config['train']
)
test_dataset = CustomDataset(
    config,
    cont_scalers=train_dataset.cont_scalers,
    cat_scalers=train_dataset.cat_scalers,
    train=~config['train']
)

#%% 
""" 2-1. 원본 데이터와 비교 """
import matplotlib.pyplot as plt

def visualize_data(idx):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(data[features[idx]], bins=30)
    ax1.set_title(f'Original: {features[idx]}')
    ax2.hist(train_dataset.data[:, idx], bins=30)
    ax2.set_title(f'Preprocessed: {features[idx]}')
    plt.tight_layout()
    plt.show()

i = 0
for idx in range(len(features)):
    visualize_data(idx)
    i += 1
    if i == 12:
        break
# %%
""" 3. Variational Autoencoder (VAE) """
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.preprocess import build_num_inverse_fn, build_cat_inverse_fn

class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()

        self.config = config
        self.device = device
        self.hidden_dim = config['hidden_dim']
        self.beta = config['beta']
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(config["input_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["latent_dim"] * 2),
        ).to(device)

        """decoder"""
        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config["input_dim"]),
        ).to(device)

    def get_posterior(self, input):
        # h = self.encoder(nn.Flatten()(input))
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.config["latent_dim"], dim=1)
        return mean, logvar

    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        noise = torch.randn(input.size(0), self.config["latent_dim"]).to(self.device)
        latent = mean + torch.exp(logvar / 2) * noise
        return mean, logvar, latent

    def forward(self, input):
        mean, logvar, latent = self.encode(input)
        xhat = self.decoder(latent)
        return mean, logvar, latent, xhat

    def loss(self, x, mean, logvar, xhat):
        recon = F.mse_loss(xhat, x, reduction='mean') / 2
        entropy = -0.5 * self.beta * (1 + logvar - mean**2 - logvar.exp()).sum(dim=1).mean()
        return recon + entropy

    def generate_synthetic_data(self, n, train_dataset):
        steps = n // self.config["batch_size"] + 1
        data = []
        with torch.no_grad():
            for _ in range(steps):
                noise = torch.randn(self.config["batch_size"], self.config["latent_dim"]).to(self.device)
                fake = self.decoder(noise)
                data.append(fake.cpu().numpy())
        data = np.concatenate(data, axis=0)[:n]

        n_cont = len(train_dataset.continuous_features)
        X_num = data[:, :n_cont]
        X_cat = data[:, n_cont:].round().astype(np.int64)

        num_inverse = build_num_inverse_fn(train_dataset.cont_scalers)
        cat_inverse = build_cat_inverse_fn(train_dataset.cat_scalers)

        syndata = pd.DataFrame(
            np.concatenate([num_inverse(X_num), cat_inverse(X_cat)], axis=1),
            columns=train_dataset.continuous_features + train_dataset.categorical_features
        )
        syndata[train_dataset.integer_features] = syndata[train_dataset.integer_features].round(0).astype(int)
        return syndata

config['batch_size'] = 64
config['hidden_dim'] = 64
config['latent_dim'] = 4
config['input_dim'] = len(features)
config['beta'] = 1.0
model = VAE(config, device)

# model.get_posterior(torch.randn(1, config['input_dim']).to(device))
# %%
""" 4. Train """
from torch.utils.data import DataLoader
from tqdm import tqdm 
config['epochs'] = 100
config['lr'] = 1e-3

dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

model.train()
for epoch in tqdm(range(config['epochs']), desc="Training VAE"):
    epoch_loss = 0
    for x_num, x_cat in dataloader:
        x = torch.cat([x_num, x_cat.float()], dim=1).to(device)

        optimizer.zero_grad()
        mean, logvar, latent, xhat = model(x)
        loss = model.loss(x, mean, logvar, xhat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(dataloader):.4f}")
# %%
""" 5. Inference """
model.eval()
n = len(train_dataset)
syndata = model.generate_synthetic_data(n, train_dataset)
print(syndata.head())
print(syndata.shape)
# %%
""" 6. Visualize """
def visualize_synthetic(idx):
    col = features[idx]
    _, ax = plt.subplots(figsize=(5, 4))
    ax.hist(train_dataset.raw_data[col], bins=30, alpha=0.5, label='Real')
    ax.hist(syndata[col], bins=30, alpha=0.5, label='Synthetic')
    ax.set_title(col)
    ax.legend()
    plt.tight_layout()
    plt.show()

for idx in range(len(features)):
    visualize_synthetic(idx)

# %%
""" 7. Beta-VAE (Scheduling Beta) """
model_beta = VAE(config, device)
optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=config['lr'])

config['epochs'] = 1000
beta = config['beta']
best_loss = float('inf')
patience = 0

model_beta.train()
for epoch in tqdm(range(config['epochs']), desc="Training Beta-VAE"):
    epoch_loss = 0
    for x_num, x_cat in dataloader:
        x = torch.cat([x_num, x_cat.float()], dim=1).to(device)

        optimizer_beta.zero_grad()
        mean, logvar, latent, xhat = model_beta(x)
        loss = model_beta.loss(x, mean, logvar, xhat)
        loss.backward()
        optimizer_beta.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience = 0
    else:
        patience += 1
        if patience >= 30:
            beta = beta * 0.7
            model_beta.beta = beta
            patience = 0
            print(f"Epoch {epoch:03d} | Beta decreased to {beta:.4f}")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Beta: {beta:.4f}")

model_beta.eval()
n = len(train_dataset)
syndata = model_beta.generate_synthetic_data(n, train_dataset)
print(syndata.head())
print(syndata.shape)

# %%
