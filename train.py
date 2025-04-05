import argparse
import os
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from HSI_dataset import Config, Data, IMG_SIZE
from models.deep_spec_sal import DeepSpectralSaliency, get_config
from utils import AverageMeter, clip_gradient, set_seed, mean_square_error, eval_smeasure


def validate(loader, net):
    """Validate the model on the test set."""
    maes = AverageMeter()
    s_measures = AverageMeter()

    net.eval()
    with torch.no_grad():
        for gt, img, (h, w), name, train_mask in loader:
            img, gt, train_mask = img.cuda(), gt.cuda(), train_mask.cuda()

            mask = (gt != -1).float()
            gt = gt * mask

            img = F.interpolate(img, IMG_SIZE, mode="nearest")

            out_final_1 = net(img)
            pred = torch.sigmoid(out_final_1)
            pred = F.interpolate(pred, (h[0], w[0]), mode="nearest")
            pred = (pred * mask.unsqueeze(1)).squeeze(1)

            mae = mean_square_error(pred, gt)
            s_mea = eval_smeasure(pred, gt)
            s_measures.update(s_mea)
            maes.update(mae)
    return maes.avg, s_measures.avg


def train(cfg, loader, val_loader, net):
    """Train the model."""
    optimizer = torch.optim.NAdam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=1e-7)
    sw = SummaryWriter()
    s_loss_record = 0.4

    for epoch in trange(cfg.epoch):
        maes = AverageMeter()
        s_meas = AverageMeter()
        losses = AverageMeter()
        net.train()

        for step, (img, gt, name, train_mask) in enumerate(loader):
            img, gt, train_mask = (
                img.type(torch.FloatTensor).cuda(),
                gt.type(torch.FloatTensor).cuda(),
                train_mask.type(torch.FloatTensor).cuda(),
            )

            mask = (gt != -1).float()
            gt = gt * mask

            result_map, loss = net(img, gt)
            result_map_ = torch.sigmoid(result_map)
            result_map_ = result_map_ * mask

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, cfg.lr)
            optimizer.step()
            losses.update(loss)

            mae_train = mean_square_error(result_map_, gt)
            s_mea = eval_smeasure(result_map_, gt)
            maes.update(mae_train)
            s_meas.update(s_mea)

        mae_loss, s_loss = validate(val_loader, net)
        sw.add_scalar("loss/valid", mae_loss, epoch + 1)
        sw.add_scalar("loss/train", maes.avg, epoch + 1)
        sw.add_scalar("loss/train_s", s_meas.avg, epoch + 1)
        sw.add_scalar("loss/valid_s", s_loss, epoch + 1)
        sw.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        scheduler.step()

        if s_loss > s_loss_record and epoch > 40:
            if not os.path.exists(cfg.savepath):
                os.makedirs(cfg.savepath)
            torch.save(net.state_dict(), os.path.join(cfg.savepath, "model-best"))
            s_loss_record = s_loss


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--data_path", type=str, default="./dataset/HRSSD")
    parser.add_argument("--save_path", type=str, default="./results/")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = get_config()
    net = DeepSpectralSaliency(config, in_channels=32).cuda()

    train_cfg = Config(
        datapath=args.data_path,
        savepath=args.save_path,
        mode="tr",
        batch=16,
        lr=3e-3,
        epoch=100,
    )
    train_data = Data(train_cfg)
    train_loader = DataLoader(
        train_data,
        collate_fn=train_data.collate,
        batch_size=train_cfg.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    test_cfg = Config(datapath=args.data_path,mode="ts")
    test_data = Data(test_cfg)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
    train(train_cfg, train_loader, test_loader, net)