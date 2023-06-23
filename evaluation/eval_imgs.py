import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import clip
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluation.base_class import IDCLIPScoreCalculator
from evaluation.clip_eval import IdCLIPEvaluator
from evaluation.parse_args import parser_eval


if __name__ == "__main__":
    """
    Usage:
    export PYTHONPATH=/gavin/code/TextualInversion/
    python evaluation/eval_imgs.py --eval_out_dir ./exp_eval/cd  \
        --eval_project_folder 2023-05-05T18-42-56_two_person-sdv4  \
        --eval_time_folder eval_2023-05-17T15-22-32
    """
    parser = argparse.ArgumentParser()
    parser = parser_eval(parser)
    opt = parser.parse_args()

    eval_folder = os.path.join(opt.eval_out_dir,
                               opt.eval_project_folder,
                               opt.eval_time_folder)

    id_clip_evaluator = IdCLIPEvaluator(
        torch.device('cuda:0'),
        clip_model='/gavin/pretrained/ViT-B-32.pt',
    )
    id_score_calculator = IDCLIPScoreCalculator(
        eval_folder,
        id_clip_evaluator,
    )

    id_score_calculator.start_calc()
