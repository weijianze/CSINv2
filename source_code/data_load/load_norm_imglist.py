import torch.utils.data as data
from PIL import Image
import os
import os.path
import pdb

def default_loader(path):
    img = Image.open(path).convert('L')
    # cropedIm = im.crop((700, 100, 1200, 1000))
    # img = Image.open(path)
    # tmpReImge = tmpImage.resize([reim_size,reim_size])
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


def multi_list_reader(root, fileList):
    imgList = []
    clsList = []
    # pdb.set_trace()
    if len(root)==len(fileList):
        for i in range(len(root)):
            cls_max = 0
            with open(fileList[i], 'r') as file:
                for line in file.readlines():
                    imgPath, label = line.strip().split(' ')
                    label = int(label)
                    # pdb.set_trace()
                    if label>cls_max:
                        cls_max = label
                    imgList.append((root[i], imgPath, sum(clsList)+label))
                    # imgList.append((os.path.join(root[i], imgPath), int(label)))
            clsList.append(cls_max+1)
    else:
        print('!!!!!')
    # pdb.set_trace()
    return imgList


class ImageList(data.Dataset):
    # def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
    def __init__(self, root, fileList, transform=None, list_reader=multi_list_reader, loader=default_loader):
        self.root      = root
        # self.imgList   = list_reader(fileList)
        self.imgList   = list_reader(root,fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        # imgPath, target = self.imgList[index]
        # img = self.loader(os.path.join(self.root, imgPath))
        img_root, imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(img_root, imgPath))
    
        if self.transform is not None:
            img = self.transform(img)
        # pdb.set_trace()
        
        img_mean = img.mean()
        img_std = img.std()+1e-8
        img = (img-img_mean)/img_std
        
        return img, target

    def __len__(self):
        return len(self.imgList)
