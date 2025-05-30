import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from evaluator import Eval_thread
from dataloader import EvalDataset
import argparse


def main(cfg):
    root_dir = cfg.root_dir
    output_dir = root_dir
    pred_dir = root_dir

    threads = []

    loader = EvalDataset(pred_dir, cfg.gt_dir)
    thread = Eval_thread(loader, cfg.method, cfg.dataset, output_dir, cfg.cuda)
    threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--dataset', type=str, default='HRSSD')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
