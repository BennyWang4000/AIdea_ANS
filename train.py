# %%
import enum
import yaml
import torch
import tqdm

from model.dataset import AudioDataset
from model.models import Generator, Discriminator

with open("config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

dataloader = torch.utils.data.DataLoader(
    datasets=AudioDataset, batch_size=config_params.get('batch_size'), suffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

# %%


def train():
    for epoch in range(config_params.epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i in enumerate(dataloader):
            progress_bar.update(1)
