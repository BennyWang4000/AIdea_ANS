# %%
from email.mime import audio
import enum
from pickletools import optimize
import yaml
import torch
import tqdm
import speechbrain as sb

from model.dataset import AudioDataset
from model.models import Generator, Discriminator, Model
from model.utils import custom_pesq, show_plt, pesq, save_flac

with open("config.yaml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)


dataset = AudioDataset(cfg.train_data_path, cfg.train_class)

train_num = dataset.__len__() * cfg.train_per
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_num, dataset.__len__() - train_num])

dataloader = torch.utils.data.DataLoader(
    datasets=AudioDataset, batch_size=cfg.get('batch_size'), suffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)
# %%

net = Model().to(device)


netG_X2Y = Generator().to(device)
netG_Y2X = Generator().to(device)
netD_X = Discriminator().to(device)
netD_Y = Discriminator().to(device)


loss_func = torch.nn.MSELoss().to(device)
# %%


g_params = list(netG_X2Y.parameters()) + \
    list(netG_Y2X.parameters())  # Get generator parameters

optimizer = torch.optim.Adam(
    net.parameters, cfg.lr, betas=(cfg.beta1, cfg.beta2))

# Create optimizers for the generators and discriminators
g_optimizer = torch.optim.Adam(g_params, cfg.lr, [cfg.beta1, cfg.beta2])

d_x_optimizer = torch.optim.Adam(
    netD_X.parameters(), cfg.lr, [cfg.beta1, cfg.beta2])
d_y_optimizer = torch.optim.Adam(
    netD_Y.parameters(), cfg.lr, [cfg.beta1, cfg.beta2])

# %%
loss_lst = []
pesq_lst = []


def train():
    for epoch in range(cfg.epochs):
        print('epoch: ', epoch)
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in enumerate(progress_bar):

            audio_data = data['audio'].to(device)
            rate = data['data'].to(device)

            g_optimizer.zero_grad()

            noise_pre = net(audio_data)
            audio_pre = audio_data - noise_pre
            pesq_soc = pesq(rate, audio_data, audio_pre)

            pesq_weight = custom_pesq(pesq_soc)

            loss = loss_func(audio_data, noise_pre) * pesq_weight
            loss.backward()

            pesq_lst.append(pesq_soc)
            loss_lst.append(loss)

            optimizer.step()
            # audio_X2Y = netG_X2Y(audio_data)
            # audio_Y2X = netG_Y2X(audio_data)

            # loss_X2Y = loss_func(audio_data, audio_X2Y)
            # loss_Y2X = loss_func(audio_data, audio_Y2X)

        show_plt('loss', loss_lst)
        show_plt('pesq', pesq_lst)
        save_flac(cfg.saving_path, str(epoch) + '.flac', audio_pre, rate)

# %%
