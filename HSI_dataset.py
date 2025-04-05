import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import trange

IMG_SIZE = 256


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, train_mask):
        for t in self.transforms:
            img, gt, train_mask = t(
                img, gt, train_mask
            )
        return img, gt, train_mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, gt, train_mask):
        if np.random.random() < 0.5:
            return (
                img[:, :, torch.arange(img.shape[2] - 1, -1, -1)],
                gt[:, torch.arange(img.shape[2] - 1, -1, -1)],
                train_mask[:, torch.arange(img.shape[2] - 1, -1, -1)],
            )
        else:
            return img, gt, train_mask


class RandomCrop(object):
    def __call__(self, img, gt, train_mask):
        H, W = gt.shape
        randw = torch.randint(W // 8 + 1, (1,)).item()  # +1确保范围不为空
        randh = torch.randint(H // 8 + 1, (1,)).item()
        offseth = 0 if randh == 0 else torch.randint(randh, (1,)).item()
        offsetw = 0 if randw == 0 else torch.randint(randw, (1,)).item()
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        img = img[:, p0:p1, p2:p3]
        gt = gt[p0:p1, p2:p3]
        train_mask = train_mask[p0:p1, p2:p3]
        return img, gt, train_mask


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.joint_transform_train = Compose(
            [
                RandomHorizontallyFlip(),
                RandomCrop(),
            ]
        )

        self.spec_transform_test = transforms.Compose(
            [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
        )

        self.image_list = []
        self.label_list = []
        self.name_list = []
        self.img_list = []
        self.gt_list = []
        self.train_mask_list = []

        data_path = os.path.join(cfg.datapath, cfg.mode, 'image')
        for _, _, fnames in sorted(os.walk(data_path)):
            for fname in fnames:
                if is_image_file(fname):
                    image_path = os.path.join(data_path, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    self.image_list.append(image_path)
                    self.label_list.append(label_path)

        for i in trange(len(self.image_list)):
            image_file = self.image_list[i]
            label_file = self.label_list[i]
            name = os.path.basename(image_file).split("/")[-1].split(".")[0]
            image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
            label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

            img = image_dataset.ReadAsArray()
            img = torch.tensor(img, dtype=torch.float) / 10000.0  # [D, H, W]

            gt = label_dataset.ReadAsArray()
            gt = torch.tensor(gt, dtype=torch.long)  # [H, W]

            mask_path = os.path.join(self.cfg.datapath, self.cfg.mode, "mask", name + ".png")
            train_mask = cv2.imread(mask_path, 0)
            train_mask = torch.tensor(train_mask, dtype=torch.float)
            self.img_list.append(img)
            self.gt_list.append(gt)
            self.train_mask_list.append(train_mask)
            self.name_list.append(name)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        gt = self.gt_list[idx]
        train_mask = self.train_mask_list[idx]
        name = self.name_list[idx]

        img = img * train_mask
        gt = gt * train_mask

        if self.cfg.mode == "tr":
            img, gt, train_mask = self.joint_transform_train(img, gt, train_mask)
            return img, gt, name, train_mask
        else:
            shape = gt.size()
            return gt, img, shape, name, train_mask

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate(batch):
        img, gt, name, train_mask = [
            list(item) for item in zip(*batch)
        ]
        for i in range(len(batch)):
            img[i] = F.interpolate(img[i].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
            gt[i] = F.interpolate(gt[i].unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
            train_mask[i] = F.interpolate(train_mask[i].unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                                          mode="nearest").squeeze(0)
        img = torch.stack(img)
        gt = torch.stack(gt)
        train_mask = torch.stack(train_mask)
        return img, gt, name, train_mask
