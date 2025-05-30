import torch
from torch.utils import data
from osgeo import gdal
import os


class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name.split('.')[0] + '.jpg' in lst_pred:
                lst.append(name.split('.')[0])

        self.image_path = list(map(lambda x: os.path.join(img_root, x + '.jpg'), lst))
        self.label_path = list(map(lambda x: os.path.join(label_root, x + '.tif'), lst))

    def __getitem__(self, item):
        label_dataset = gdal.Open(self.label_path[item], gdal.GA_ReadOnly)
        pred = gdal.Open(self.image_path[item], gdal.GA_ReadOnly)

        gt = label_dataset.ReadAsArray()
        gt = torch.tensor(gt, dtype=torch.float32)

        pred = pred.ReadAsArray()
        pred = torch.tensor(pred, dtype=torch.float32)
        pred = pred / 255.

        mask = (gt != -1)
        mask = torch.tensor(mask, dtype=torch.float32)

        return pred, gt, mask

    def __len__(self):
        return len(self.image_path)
