import os
import time
import datetime
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from torch import autocast
from torchvision.transforms import transforms
from pytorch_lightning import seed_everything

from PIL import Image
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from evaluation.base_class import ModelInferBase, EvalDatasetBase, ImageSynthesizerBase
from evaluation.parse_args import parser_main, parser_gen
from evaluation.prompt_templates import get_pos_neg_temps


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class ModelInferOurs(ModelInferBase):
    def _load_model(self, opt) -> nn.Module:

        project = opt.eval_project_folder.replace('_textualinversion', '')
        opt.config = f"logs/{opt.eval_project_folder}/configs/{project}-project.yaml"
        opt.embedding_path = f"logs/{opt.eval_project_folder}/checkpoints/" \
                             f"embeddings_gs-{opt.eval_step}.pt"
        opt.img_suffix = f"{opt.img_suffix}_{opt.eval_id}_{opt.eval_step}_img{opt.eval_img_idx}"

        config = OmegaConf.load(f"{opt.config}")
        config.model.params.personalization_config.params.use_saved_id = not opt.eval_not_use_saved_id
        model = load_model_from_config(config, f"{opt.ckpt}")
        model.embedding_manager.load(opt.embedding_path)
        model = model.to(self.device)

        sampler = DDIMSampler(model)
        self.sampler = sampler

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        self.start_code = start_code

        self.temp_pos, self.temp_neg = get_pos_neg_temps(opt.from_file)

        return model

    def infer_one(self, prompt: list, in_image: torch.Tensor, in_ids: torch.Tensor, nums_id: torch.Tensor):
        opt = self.opt
        model = self.model
        batch_size = self.batch_size
        sampler = self.sampler
        start_code = self.start_code

        image_ori = {
            "faces": in_image,  # not always use_saved_id mode
            "ids": in_ids,
            "num_ids": nums_id
        }
        # prompt = [self.temp_pos.format(p) for p in prompt]  # use long prompt

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        neg_prompt = self.temp_neg
                        uc = model.get_learned_conditioning(batch_size * [neg_prompt])
                    if isinstance(batch_size, str):
                        batch_prompts = [prompt] * batch_size
                    else:
                        batch_prompts = prompt
                    c = model.get_learned_conditioning(batch_prompts, image_ori=image_ori)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)

        return x_samples_ddim


class ModelInferTI(ModelInferBase):
    def _load_model(self, opt) -> nn.Module:

        project = opt.eval_project_folder.replace('_textualinversion', '')
        opt.config = "./configs/stable-diffusion/v1-inference.yaml"
        opt.embedding_path = f"logs/{opt.eval_project_folder}/checkpoints/" \
                             f"embeddings_gs-{opt.eval_step}.pt"
        opt.img_suffix = f"{opt.img_suffix}_{opt.eval_id}_{opt.eval_step}_img{opt.eval_img_idx}"

        config = OmegaConf.load(f"{opt.config}")
        model = load_model_from_config(config, f"{opt.ckpt}")
        model.embedding_manager.load(opt.embedding_path)
        model = model.to(self.device)

        sampler = DDIMSampler(model)
        self.sampler = sampler

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        self.start_code = start_code

        return model

    def infer_one(self, prompt: str, in_image: torch.Tensor, in_ids: torch.Tensor, nums_id: torch.Tensor):
        opt = self.opt
        model = self.model
        batch_size = self.batch_size
        sampler = self.sampler
        start_code = self.start_code

        # image_ori = {
        #     "faces": in_image,
        #     "ids": in_ids,
        #     "num_ids": nums_id
        # }

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(batch_size, str):
                        batch_prompts = [prompt] * batch_size
                    else:
                        batch_prompts = prompt
                    c = model.get_learned_conditioning(batch_prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)

        return x_samples_ddim


if __name__ == "__main__":
    """
    Usage:
    export PYTHONPATH=/gavin/code/TextualInversion/
    CUDA_VISIBLE_DEVICES=0 python evaluation/gen_imgs.py --eval_out_dir exp_eval/ours/  \
        --eval_project_folder training2023-05-24T16-41-49_textualinversion  \
        --eval_step 1499  \
        --eval_dataset st7  \
        --eval_id 1  \
        --eval_id2 6  \
        --from-file ./infer_images/tmp.txt  \
        --n_iter 1
    """
    parser = argparse.ArgumentParser()
    parser = parser_main(parser)
    parser = parser_gen(parser)
    opt = parser.parse_args()

    eval_dataset = EvalDatasetBase(
        opt.eval_dataset, opt.eval_id, opt.from_file, opt.eval_img_idx,
        eval_id2s=opt.eval_id2,
    )

    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_to_folder = f'eval_{now}'

    if len(opt.eval_resume_folder) > 0:
        save_to_folder = opt.eval_resume_folder

    if 'ours' in opt.eval_out_dir:
        generator = ModelInferOurs(
            os.path.join(opt.eval_out_dir, opt.eval_project_folder, save_to_folder),
            opt.n_samples,
            device,
            opt,
            repeats=opt.n_iter,
            resume_cnt=opt.eval_resume_cnt,
        )
    elif 'ti' in opt.eval_out_dir:
        generator = ModelInferTI(
            os.path.join(opt.eval_out_dir, opt.eval_project_folder, save_to_folder),
            opt.n_samples,
            device,
            opt,
            repeats=opt.n_iter,
            resume_cnt=opt.eval_resume_cnt,
        )
    else:
        raise ValueError()

    synthesizer = ImageSynthesizerBase(
        eval_dataset,
        generator,
    )
    synthesizer.start_synthesize()
