# %%
import os
import glob
import shutil
from tqdm import tqdm

DATA_PATH = 'D:\\CodeRepositories\\py_project\\aidea\\ANS\\data\\train_classification\\mixed'
VOCAL_PATH = 'D:\\CodeRepositories\\py_project\\aidea\\ANS\\data\\train'
DST_PATH = 'G:\\我的雲端硬碟\\Code Repository\\aidea\\ANS\\data\\train_classification\\vocal'
# DST_PATH = 'D:\\CodeRepositories\\py_project\\aidea\\ANS\\data\\train_classification\\vocal'

vocal_lst = sorted(glob.glob(os.path.join(VOCAL_PATH, 'vocal*.*')))
class_lst = sorted(os.listdir(DATA_PATH))

print(class_lst)
# %%
for classname in class_lst:
    print(classname)
    dst_path = os.path.join(DST_PATH, classname)
    mixed_lst = sorted(glob.glob(os.path.join(
        DATA_PATH, classname, 'mixed*.*')))

    number_lst = [filepath.split('\\')[-1][6:11] for filepath in mixed_lst]

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for number in tqdm(number_lst, total=len(number_lst)):
        filename = 'vocal_' + str(number) + '.flac'
        shutil.copy(os.path.join(VOCAL_PATH, filename),
                    os.path.join(dst_path, filename))
print('done')
# %%
