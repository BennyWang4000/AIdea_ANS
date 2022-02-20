from torch.utils.data import Dataset
import os
import soundfile as sf
import numpy as np


class AudioDataset(Dataset):

    def __init__(self, transform, root, data_class, signal_bound):
        super(AudioDataset, self).__init__()
        self.transform = transform
        self.signal_bound = signal_bound
        self.root = root
        self.data_class = data_class
        # self.train_data = sorted(glob.glob(os.path.join(root, "/*.*")))
        self.list = os.listdir(os.path.join(self.root, data_class))
        self.data_len = len(self.list)

    def __getitem__(self, index):
        # os.path.join(self.root, self.list[index])
        zero = np.zeros((self.signal_bound,))

        data, rate = sf.read(os.path.join(
            self.root, self.data_class, self.list[index]))
        if data.shape[0] > self.signal_bound:
            zero = data[:self.signal_bound]
        else:
            zero[:data.shape[0]] = data

        # zero[:data.shape[0]] = data

        # sample= self.transform({'audio': zero, 'rate': rate})

        return zero, rate
        # return np.expand_dims(zero, 0), rate

    def __len__(self):
        return self.data_len
