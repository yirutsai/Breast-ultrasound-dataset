import shutil
from pathlib import Path

import os

pwd = Path(os.getcwd())

data_root = Path(os.path.dirname(__file__))/Path("data")
os.makedirs(data_root)

for sdir in os.listdir(pwd/Path("Dataset_BUSI_with_GT")):
    print(sdir)
    os.makedirs(data_root/Path(sdir)/Path("images"),exist_ok=True)
    os.makedirs(data_root/Path(sdir)/Path("mask"),exist_ok=True)

    for file_name in os.listdir(pwd/Path("Dataset_BUSI_with_GT")/Path(sdir)):
        if("mask" in file_name):
            shutil.copyfile(pwd/Path("Dataset_BUSI_with_GT")/Path(sdir)/Path(file_name),data_root/Path(sdir)/Path("mask")/Path(file_name))
        else:
            shutil.copyfile(pwd/Path("Dataset_BUSI_with_GT")/Path(sdir)/Path(file_name),data_root/Path(sdir)/Path("images")/Path(file_name))
        