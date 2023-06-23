import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    _, extension = os.path.splitext(ckpt)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/aigc_id_infer.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--save_fp16",
        action="store_true",
        help="If stored, fp16=True. (default: False)")
    parser.add_argument(
        "--save_celeb_basis_path",
        type=str,
        default="./weights/celeb_basis.pt",
        help="Where to save celeb basis.")
    parser.add_argument(
        "--save_ti_embedding_save_folder",
        type=str,
        default="./weights/ti_id_embeddings",
        help="Where to save ID embeddings.")

    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    ''' load trained params '''
    celeb_basis_path = opt.save_celeb_basis_path
    model.cond_stage_model.save_celeb_embeddings(celeb_basis_path)

    celeb_basis = model.cond_stage_model.celeb_embeddings
    c_mean, pca_base = celeb_basis[:, 0], celeb_basis[:, 1:]
    c_mean = c_mean.unsqueeze(1)  # mean:(es,1,768)

    id_coefficients = model.embedding_manager.id_coefficients
    id_embeddings = model.embedding_manager.id_embeddings
    max_ids = len(id_coefficients)
    for idx in range(max_ids):
        id_coefficients[idx] = id_coefficients[idx].to(device)

    ''' extract, calculate, and save '''
    es, h, inner_dim = id_coefficients[0].shape
    es, inner_dim, k = pca_base.shape

    ti_embeddings = [torch.zeros(es, k, dtype=id_coefficients[0].dtype).to(device)] * max_ids
    for idx in range(max_ids):
        x = id_coefficients[idx]
        z = torch.einsum('e h k, e k c -> e h c', x, pca_base) + c_mean  # (es,h,768)
        z = rearrange(z, 'e h c -> (e h) c').contiguous()  # (es*h,768)
        ti_embeddings[idx] = z

    ti_embedding_save_folder = opt.save_ti_embedding_save_folder
    os.makedirs(ti_embedding_save_folder, exist_ok=True)
    for idx in range(max_ids):
        if opt.save_fp16:
            torch.save(ti_embeddings[idx].cpu().half(),
                       os.path.join(ti_embedding_save_folder,
                                    "id_embedding_fp16_{:03d}.pt".format(idx)))
            torch.save(id_coefficients[idx].cpu().half(),
                       os.path.join(ti_embedding_save_folder,
                                    "id_coefficient_fp16_{:03d}.pt".format(idx)))
        else:
            torch.save(ti_embeddings[idx].cpu(),
                       os.path.join(ti_embedding_save_folder,
                                    "id_embedding_{:03d}.pt".format(idx)))
            torch.save(id_coefficients[idx].cpu(),
                       os.path.join(ti_embedding_save_folder,
                                    "id_coefficient_{:03d}.pt".format(idx)))

    print(f"TI ID embeddings saved to {ti_embedding_save_folder}, "
          f"shape={ti_embeddings[0].shape}")


if __name__ == "__main__":
    main()
