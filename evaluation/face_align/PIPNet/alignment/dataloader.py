import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image

from evaluation.face_align.PIPNet.alignment.alignment import norm_crop
from evaluation.face_align.PIPNet.alignment.landmarks import get_5_from_98
from evaluation.face_align.PIPNet.lib.tools import get_lmk_model, demo_image


class FolderAlignCrop(Dataset):
    def __init__(self,
                 folder_path: str,
                 image_size: int = 512,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 align_mode: str = 'ffhq',
                 ):
        super(FolderAlignCrop, self).__init__()

        self.root = folder_path
        self.imgs = os.listdir(folder_path)

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transform
        ])
        self.align_mode = align_mode

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        print('alignment model loaded')

    def __getitem__(self, index):
        t_id = index

        t_img, has_lmk_box = self._check_lmk_box(t_id)

        if self.transform is not None:
            t_img = self.transform(t_img)

        return {
            "target_image": t_img,
            "has_lmk_box": has_lmk_box,
            "img_name": self.imgs[t_id]
        }

    def __len__(self):
        return len(self.imgs)

    def _check_lmk_box(self, t_id):
        img_path = os.path.join(self.root, self.imgs[t_id])
        full_img = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)

        ''' face alignment and check landmarks '''
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, 256, mode=self.align_mode, borderValue=0.0)
            return cropped_img, True
        else:
            print('Landmarks checking failed @ (id=%d)' % t_id)
            return full_img, False
