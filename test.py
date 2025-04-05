import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from HSI_dataset import Data, Config, IMG_SIZE
from models.deep_spec_sal import DeepSpectralSaliency, get_config

save_path = "DataStorage/test_result"


class Test(object):
    def __init__(self, Network, Path, snapshot):
        ## dataset
        self.cfg = Config(datapath=Path, snapshot=snapshot, mode="ts")

        self.config = get_config()
        self.net = Network(self.config).cuda()

        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(
            self.cfg.snapshot, map_location=torch.device("cpu")
        )
        pretrained_dict = {
            k.replace("module.", ""): v
            for k, v in pretrained_dict.items()
            if (k.replace("module.", "") in model_dict)
        }

        # check unloaded weights
        for k, v in model_dict.items():
            if k in pretrained_dict.keys():
                pass
            else:
                print("miss keys in pretrained_dict: {}".format(k))

        model_dict.update(pretrained_dict)
        print("load pretrained model from {}".format(self.cfg.snapshot))
        self.net.load_state_dict(model_dict)

        self.net.train(False)

        self.data = Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=8, shuffle=False, num_workers=0)

    def save(self):
        with torch.no_grad():
            for gt, img, (H, W), name, train_mask in self.loader:
                img = img.cuda().float()
                gt = gt.cuda().float()
                mask = (gt != -1).float()
                img = F.interpolate(img, IMG_SIZE, mode="nearest")

                out_final_1 = self.net(img)

                pred = torch.sigmoid(out_final_1)
                pred = F.interpolate(pred, (H[0], W[0]), mode="nearest")

                for i in range(pred.shape[0]):
                    cv2.imwrite(
                        save_path + "/" + name[i].split(".")[0] + ".jpg", pred[i].squeeze(0).cpu().numpy() * 255
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--model_path", type=str, default="./results/model-best")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    t = Test(
        DeepSpectralSaliency,
        "./dataset/HRSSD",
        args.model_path,
    )
    t.save()
