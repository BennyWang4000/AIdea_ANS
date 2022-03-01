from torch import device, cuda

# * data config =============
TRAIN_DATA_PATH = '/gdrive/MyDrive/Code Repository/aidea/ANS/data/train_classification'
TEST_DATA_PATH = '/gdrive/MyDrive/Code Repository/aidea/ANS/data/test'
SAVING_PATH = '/gdrive/MyDrive/Code Repository/aidea/ANS/AIdea_ANS/runs'
TRAIN_CLASS = 'dog_bark'

# train_data_path= 'D:/CodeRepositories/py_project/aidea/ANS/data/train/'
# test_data_path= 'D:/CodeRepositories/py_project/aidea/ANS/data/test'

# * train config =============
TRAIN_PER = 0.9
IS_VALID = True
CHECK_POINT= 10

EPOSIDE = 5
EPOCHS = 2
BATCH_SIZE = 10
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
GAMMA = 0.99


# * signal config ============
SIGNAL_BOUND = 99000
N_FFT = 256
HOP_LENGTH = 8

STFT_SHAPE = (129, 12376)  # magic number
SAMPLE_RATE = 16000

# * torch init ===============
IS_CUDA = cuda.is_available()
DEV = device("cuda:0" if cuda.is_available() else "cpu")
