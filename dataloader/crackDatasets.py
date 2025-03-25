import os
import random

from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from dataloader.augment import get_train_augmentation


def readpath(root,txt_path):
    img_list = []
    txt_path=os.path.join(root,txt_path)
    with open(txt_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            tmp=[]
            tmp.append(os.path.join(root,item[0]))
            tmp.append(os.path.join(root,item[1]))
            img_list.append(tmp)
    file_to_read.close()
    return img_list
class crackDataset(data.Dataset):
    def __init__(self, root,txt,imgsize=512):
        super().__init__()
        self.root=root
        self.txt=txt
        self.imgsize=imgsize
        self.pathlist = readpath(self.root,txt)
        self.train_transforms=get_train_augmentation((512,512))
        self.normal_trans=T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath=self.pathlist[idx][0]
        maskpath=self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')
        image=self.normal_trans(image)
        mask=self.normal_trans(mask)

        if self.txt=="train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1
        return (image, mask)

class crackDataset_withname(data.Dataset):
    def __init__(self, root,txt,imgsize=512):
        super().__init__()
        self.root=root
        self.txt=txt
        self.imgsize=imgsize
        self.pathlist = readpath(self.root,txt)
        self.train_transforms=get_train_augmentation((512,512))
        self.normal_trans=T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath=self.pathlist[idx][0]
        maskpath=self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')
        image=self.normal_trans(image)
        mask=self.normal_trans(mask)

        if self.txt=="train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1
        return (image, mask,imagepath)