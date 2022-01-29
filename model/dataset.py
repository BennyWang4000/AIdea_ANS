from torch.utils.data import Dataset
from torch import load
import os
import glob
import soundfile as sf


class AudioDataset(Dataset):
    """Some Information about NoiseDataset"""

    def __init__(self, root, data_class):
        # super(AudioDataset, self).__init__()
        # self.root = 'D:/CodeRepositories/py_project/aidea/ANS/data/train/'
        self.root = root
        self.data_class = data_class
        # self.train_data = sorted(glob.glob(os.path.join(root, "/*.*")))
        self.list = os.listdir(os.path.join(self.root, data_class))

    def __getitem__(self, index):
        # os.path.join(self.root, self.list[index])
        data, rate = sf.read(os.path.join(
            self.root, self.data_class, self.list[list]))
        return data, rate

    def __len__(self):
        return len(self.list)
