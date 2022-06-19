import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from skimage import data, io, filters

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchsummary import summary
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import cv2

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--seed",type =int, default = 3030)
parser.add_argument("--batch_size",type = int, default= 4)
parser.add_argument("--loss_type",type=str,default="dice")
parser.add_argument("--output_dir",type =str,default="output")

args = parser.parse_args()



# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.2),(0.6)),
])
pwd = Path(os.getcwd())

data_root = Path(os.path.dirname(__file__))/Path("data")


labeldict = {"benign":0,"malignant":1,"normal":2}
idx2label = {0:"benign",1:"malignant",2:"normal"}
def load_data():
    images = []
    labels = []
    masks = []
    for sdir in os.listdir(data_root):
        print(sdir)
        for file_name in os.listdir(data_root/Path(sdir)/Path("images")):
            labels.append(labeldict[sdir])
            images.append(test_tfm(cv2.imread(str(data_root/Path(sdir)/Path("images")/Path(file_name)))))
            masks.append(test_tfm(cv2.imread(str(data_root/Path(sdir)/Path("mask")/Path(file_name.rstrip(".png")+"_mask.png")))))
            # print(images[0])
            # print(masks[0])
            # print(np.array(Image.open(data_root/Path(sdir)/Path("images")/Path(file_name)).getdata()))
            # print(Image.open(data_root/Path(sdir)/Path("mask")/Path(file_name.rstrip(".png")+"_mask.png")))
            # exit()
    assert len(images) == len(labels) ==len(masks) == 780
    return labels,images,masks


class UltraSoundDataset(Dataset):
    def __init__(self,images,masks,labels,transform):
        super().__init__()
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        return self.images[index],self.masks[index],self.labels[index]
    def __len__(self):
        assert len(images) == len(labels) ==len(masks)
        return len(self.images)
    
myseed = args.seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

batch_size = args.batch_size

print("Constructing Dataset...")
if(os.path.exists("train_dataset.pkl")==False or os.path.exists("test_dataset.pkl")==False):
    print("Building dataset")
    labels,images,masks = load_data()
    dataset = UltraSoundDataset(images,masks,labels,test_tfm)
    # torch.save(dataset,"dataset.data")
    train_dataset,test_dataset = train_test_split(dataset,test_size=0.2)
else:
    print("Loading dataset from train_dataset.pkl and test_dataset.pkl")
    with open("train_dataset.pkl","rb")as f:
        train_dataset = pickle.load(f)
    with open("test_dataset.pkl","rb")as f:
        test_dataset = pickle.load(f)

train_dataset_pred = np.load("train_pred.npy")
test_dataset_pred = np.load("test_pred.npy")

for i in range(len(train_dataset)):
    train_dataset[i] = tuple(list(train_dataset[i])+[train_dataset_pred[i]])
for i in range(len(test_dataset)):
    test_dataset[i] = tuple(list(test_dataset[i])+[test_dataset_pred[i]])
detect_train_acc = detect_test_acc = 0
for i in range(len(train_dataset)):
    if((train_dataset[i][2]==2)==(train_dataset[i][3]==2)):
        detect_train_acc +=1
for i in range(len(test_dataset)):
    if((test_dataset[i][2]==2)==(test_dataset[i][3]==2)):
        detect_test_acc +=1

print(f"detect_train_acc: {detect_train_acc/len(train_dataset)}")
print(f"detect_test_acc: {detect_test_acc/len(test_dataset)}")
print(len(train_dataset))
# print(len(valid_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle = False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle = False)

device = "cuda" if torch.cuda.is_available() else "cpu"
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        self.sigmoid = nn.Sigmoid()
        return self.sigmoid(self.conv(x))
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):    
        # inputs = torch.sigmoid(inputs)      
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice
if(args.loss_type=="dice"):
    criterion = DiceLoss()
elif(args.loss_type =="BCE"):
    print("using BCE")
    criterion = nn.BCELoss()
elif(args.loss_type =="MSE"):
    criterion = nn.MSELoss()


best_loss = float("inf")
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
model = torch.load(f"./model_seg_base_{args.loss_type}.pt")
model2 = torch.load(f"./model_aid_seg_{args.loss_type}.pt")
for key in labeldict:
    os.makedirs(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{key}"),exist_ok=True)
    os.makedirs(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{key}"),exist_ok=True)
for idx,(img, mask , label,pred_label) in enumerate(train_loader):
    img,mask = img.to(device),mask.to(device)
    with torch.no_grad():
        output = model(img).permute(0,2,3,1).detach().cpu().numpy()*255
        aid_output = model2(img).permute(0,2,3,1).detach().cpu().numpy()*255
    idxx = pred_label==2
    aid_output[idxx,:] = 0
    label = label.numpy()
    img = img.permute(0,2, 3, 1).cpu().numpy()*255
    mask = mask.permute(0,2,3,1).cpu().numpy()*255
    for i in range(batch_size):
        # print(img[i])
        # print(img[i].shape)
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{idx2label[label[i]]}/train-{idx*batch_size+i}-img.png")),img[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{idx2label[label[i]]}/train-{idx*batch_size+i}-mask.png")),mask[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{idx2label[label[i]]}/train-{idx*batch_size+i}-pred.png")),output[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{idx2label[label[i]]}/train-{idx*batch_size+i}-pred_aid.png")),aid_output[i])

for idx,(img, mask , label,pred_label) in enumerate(test_loader):
    img,mask = img.to(device),mask.to(device)
    with torch.no_grad():
        output = model(img).permute(0,2,3,1).detach().cpu().numpy()*255
        aid_output = model2(img).permute(0,2,3,1).detach().cpu().numpy()*255
    idxx = pred_label==2
    aid_output[idxx,:] = 0
    label = label.numpy()
    img = img.permute(0,2, 3, 1).cpu().numpy()*255
    mask = mask.permute(0,2, 3, 1).cpu().numpy()*255
    for i in range(batch_size):
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{idx2label[label[i]]}/test-{idx*batch_size+i}-img.png")),img[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{idx2label[label[i]]}/test-{idx*batch_size+i}-mask.png")),mask[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{idx2label[label[i]]}/test-{idx*batch_size+i}-pred.png")),output[i])
        cv2.imwrite(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{idx2label[label[i]]}/test-{idx*batch_size+i}-pred_aid.png")),aid_output[i])