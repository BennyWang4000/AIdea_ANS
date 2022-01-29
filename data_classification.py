# %%
import os
import glob
import shutil
from tqdm import tqdm

DATA_PATH = 'D:\\CodeRepositories\\py_project\\aidea\\ANS\\data\\train'
DST_PATH = 'D:\\CodeRepositories\\py_project\\aidea\\ANS\\data\\train_classification'

mixed_list = sorted(glob.glob(os.path.join(DATA_PATH, 'mixed*.*')))
# list = os.listdir(DATA_PATH)
class_dict = {}
progress = tqdm(total=len(mixed_list))

for filepath in mixed_list:
    filename = filepath.split('\\')[-1]
    classname = filename[12:-5]
    dst_filename = os.path.join(DST_PATH, classname)

    if classname not in class_dict:
        class_dict[classname] = 0
    else:
        class_dict[classname] = class_dict[classname] + 1

    if not os.path.exists(dst_filename):
        os.mkdir(dst_filename)

    shutil.copy(filepath, os.path.join(dst_filename, filename))

    progress.update(1)
print(class_dict)

# %%
