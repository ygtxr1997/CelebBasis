import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F

import random
import pickle

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    # 'a photo of a {} with {}',
    # 'a rendering of a {} with {}',
    # 'a cropped photo of the {} with {}',
    # 'the photo of a {} with {}',
    # 'a photo of a clean {} with {}',
    # 'a photo of a dirty {} with {}',
    # 'a dark photo of the {} with {}',
    # 'a photo of my {} with {}',
    # 'a photo of the cool {} with {}',
    # 'a close-up photo of a {} with {}',
    # 'a bright photo of the {} with {}',
    # 'a cropped photo of a {} with {}',
    # 'a photo of the {} with {}',
    # 'a good photo of the {} with {}',
    # 'a photo of one {} with {}',
    # 'a close-up photo of the {} with {}',
    # 'a rendition of the {} with {}',
    # 'a photo of the clean {} with {}',
    # 'a rendition of a {} with {}',
    # 'a photo of a nice {} with {}',
    # 'a good photo of a {} with {}',
    # 'a photo of the nice {} with {}',
    # 'a photo of the small {} with {}',
    # 'a photo of the weird {} with {}',
    # 'a photo of the large {} with {}',
    # 'a photo of a cool {} with {}',
    # 'a photo of a small {} with {}',
    'a photo of a {} and a {}',
    'a rendering of a {} and a {}',
    'a cropped photo of the {} and a {}',
    'the photo of a {} and a {}',
    'a photo of a clean {} and a {}',
    'a photo of a dirty {} and a {}',
    'a dark photo of the {} and a {}',
    'a photo of my {} and a {}',
    'a photo of the cool {} and a {}',
    'a close-up photo of a {} and a {}',
    'a bright photo of the {} and a {}',
    'a cropped photo of a {} and a {}',
    'a photo of the {} and a {}',
    'a good photo of the {} and a {}',
    'a photo of one {} and a {}',
    'a close-up photo of the {} and a {}',
    'a rendition of the {} and a {}',
    'a photo of the clean {} and a {}',
    'a rendition of a {} and a {}',
    'a photo of a nice {} and a {}',
    'a good photo of a {} and a {}',
    'a photo of the nice {} and a {}',
    'a photo of the small {} and a {}',
    'a photo of the weird {} and a {}',
    'a photo of the large {} and a {}',
    'a photo of a cool {} and a {}',
    'a photo of a small {} and a {}',
]

per_img_token_list = ['sks', 'ks', 'ata', 'tre', 'ry', 'bop', 'rn', '&', '*', '`']
reg_token_list = ['face']


class BackgroundDataset(Dataset):
    def __init__(self,
                 pickle_path: str = '/gavin/datasets/ADE20K_2021_17_01/index_ade20k.pkl',
                 root_folder: str = '/gavin/datasets/',
                 split: str = 'all',
                 out_size: int = 512,
                 ):
        self.pickle_path = pickle_path
        with open(self.pickle_path, "rb") as handle:
            pickle_dict = pickle.load(handle)
        p_filenames = pickle_dict["filename"]
        p_folders = pickle_dict["folder"]

        self.img_list = []
        for i in range(len(p_filenames)):
            if split == 'train' and 'training' not in p_folders[i]:
                continue
            elif split == 'val' and 'validation' not in p_folders[i]:
                continue
            img_path = os.path.join(root_folder, p_folders[i], p_filenames[i])
            self.img_list.append(img_path)
        print('[BackgroundDataset] loaded from %s, split=%s, total=%d.' % (
            pickle_path, split, self.__len__()))

        self.trans = transforms.Compose([
            transforms.RandomResizedCrop(out_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        bg_img = Image.open(self.img_list[index]).convert('RGB')
        bg_img = self.trans(bg_img)  # (C,H,W)
        return bg_img


class FaceIdDatasetVGG(Dataset):
    def __init__(self,
                 pickle_path: str = '/gavin/datasets/original/image_512_quality.pickle',
                 num_ids: int = 10,
                 specific_ids: list = None,
                 images_per_id: int = 10,
                 image_size: int = 512,
                 repeats: int = 100,
                 flip_p: float = 0.5,
                 split: str = "train",
                 reg_ids: int = 1000,
                 reg_images_per_id: int = 1,
                 reg_repeats: int = 10,
                 asian: bool = False,
                 diff_cnt: int = 32,
                 ):
        """ """
        super(FaceIdDatasetVGG, self).__init__()

        if isinstance(specific_ids, str):
            import re
            if re.match(r'\d+-\d+', specific_ids) is not None:
                lo, hi = [int(x) for x in specific_ids.split('-')]
                specific_ids = list(np.arange(lo, hi))
            else:
                raise ValueError('Specific_ids not supported.')

        ''' full image list '''
        self.pickle_path = pickle_path
        self.num_ids = num_ids
        self.specific_ids = specific_ids
        self.images_per_id = images_per_id
        self.reg_ids = reg_ids
        self.reg_images_per_id = reg_images_per_id
        self.asian = asian
        self.diff_cnt = diff_cnt

        self.img_dict = {}  # {'id':str, 'images':[img]}
        self.img_list = []  # [img], all images
        self._load_from_pickle()

        self.repeats = repeats
        self.reg_repeats = reg_repeats
        self.split = split
        self.img_list = self.img_list[:num_ids * images_per_id] * repeats \
                        + self.img_list[num_ids * images_per_id:] * reg_repeats

        self.num_images = len(self.img_list)
        self.num_train = num_ids * images_per_id * repeats
        self.num_reg = reg_ids * self.reg_images_per_id * reg_repeats
        self._length = self.num_images

        ''' background '''
        # self.bg_dataset = BackgroundDataset(split='val')
        # self.bg_len = len(self.bg_dataset)

        ''' transform '''
        self.image_size = image_size
        self.trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.split == 'dev':
            self.trans = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print('[FaceIdDataset] loaded from %s. (id*max_img=%d*%d, train%d+reg%d=total%d)' % (
            pickle_path,
            len(self.img_dict.keys()), len(self.img_dict[num_ids - 1]['images']),
            self.num_train, self.num_reg, self.num_images))

    def _load_from_pickle(self):
        """ """
        ''' 0. load all images '''
        with open(self.pickle_path, "rb") as handle:
            pickle_list = pickle.load(handle)
        pickle_list = sorted(pickle_list, key=lambda x: x[1])[::-1]
        pickle_dict = {}  # {id, [img]}
        for img, iqa in pickle_list:
            id = img.split("/")[-2]
            if id not in pickle_dict:
                pickle_dict[id] = [img]
            else:
                pickle_dict[id] += [img]

        ''' 1. load train '''
        iterate_id = 0  # walked id
        idx = 0  # used id
        train_id_set = set()
        for id, images in pickle_dict.items():
            if self.asian and ('n' in id):
                continue  # if vggface2 ('nxx'), skip; only keeps celeb-asian
            if len(self.img_dict.keys()) >= self.num_ids:
                break  # enough
            # if len(images) >= self.images_per_id:
            if len(images) >= 10:  # TODO: least nums
                if self.specific_ids is not None and iterate_id not in self.specific_ids:
                    iterate_id += 1
                    continue
                self.img_dict[idx] = {'id': id, 'images': images[:self.images_per_id]}
                self.img_list += images[:self.images_per_id]
                iterate_id += 1
                idx += 1
                train_id_set.add(id)
            if id == list(pickle_dict.keys())[-1]:
                raise ValueError('Reach last. Not enough images for num_ids=%d, only %d.' % (
                    self.num_ids, len(train_id_set)))

        ''' 2. load reg '''
        reg_cnt = 0
        for id, images in pickle_dict.items():
            if len(self.img_dict.keys()) >= self.num_ids + self.reg_ids:
                break  # enough
            if id in train_id_set:
                continue  # Repeat with train
            if len(images) >= self.reg_images_per_id:
                self.img_dict[idx] = {'id': id, 'images': [images[0]]}
                self.img_list += [images[0]]
                idx += 1
                per_img_token_list.append('face')
                reg_cnt += 1
            if id == list(pickle_dict.keys())[-1]:
                raise ValueError('Reach last. Not enough images for reg_ids=%d, only %d.' % (
                    self.reg_ids, reg_cnt))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        ''' image '''
        id_idx, img_idx = self._get_id_and_img_idx(i)
        img = Image.open(self.img_dict[id_idx]['images'][img_idx]).convert('RGB')
        # equals to 'self.img_list[i]'
        img = self.trans(img)  # (C,H,W)

        ''' diff id '''
        img2_list, id_idx2_list = self._get_diff_id_multi(id_idx, img_idx, cnt=self.diff_cnt)

        ''' aug2 id '''
        img3_list, id_idx3_list = self._get_aug2_id_multi([id_idx] + id_idx2_list)

        dual_img = False if np.random.randint(10) < 50 else True
        if dual_img:  # (1,diff,1+diff)
            example["image_ori"] = {'faces': torch.cat([img.permute(1, 2, 0)]
                                                       + [img2.permute(1, 2, 0) for img2 in img2_list]
                                                       + [img3.permute(1, 2, 0) for img3 in img3_list],
                                                       dim=-1),  # (N,H,W,(1+diff+same)C)
                                    'ids': torch.tensor([id_idx] + id_idx2_list + id_idx3_list),  # (N,1+diff+same)
                                    'num_ids': 2}
            img = self._mix_two_tensors(img, img2_list[0], None)
        else:  # (1,diff,1+diff)
            example["image_ori"] = {'faces': torch.cat([img.permute(1, 2, 0)]
                                                       + [img2.permute(1, 2, 0) for img2 in img2_list]
                                                       + [img3.permute(1, 2, 0) for img3 in img3_list],
                                                       dim=-1),
                                    'ids': torch.tensor([id_idx] + id_idx2_list + id_idx3_list),
                                    'num_ids': 1}
            img = self._add_bg(img, None) if not self.split == 'dev' else img

        img = img.permute(1, 2, 0)  # in [-1,1], from (C,H,W) to (H,W,C)
        example["image"] = img
        example["id_idx"] = id_idx  # not used
        example["img_idx"] = img_idx  # not used

        ''' text '''
        placeholder_string = per_img_token_list[0]
        text = random.choice(imagenet_templates_small).format('face of %s person' % placeholder_string)
        if dual_img:
            str1, str2 = per_img_token_list[0], per_img_token_list[1]
            text = random.choice(imagenet_dual_templates_small).format('face of %s person' % str1,
                                                                       'face of %s person' % str2)
            text = '{}, left is a face of {} person, right is another face of {} person'.format(text, str1, str2)
        example["caption"] = text
        return example

    def _get_id_and_img_idx(self, i):
        if i < self.num_train:
            i %= (self.num_ids * self.images_per_id)
            id_idx = i // self.images_per_id
            img_idx = i % self.images_per_id
        else:
            i -= self.num_train
            i %= (self.reg_ids * 1)
            id_idx = i // self.reg_images_per_id + self.num_ids
            img_idx = i % self.reg_images_per_id
        return id_idx, img_idx

    def _get_diff_id_multi(self, id_idx1, img_idx1, cnt: int = 16):
        assert cnt < self.num_ids
        ret_id_idx2, ret_img2 = [], []
        for _ in range(cnt):
            id_idx2, img_idx2 = id_idx1, img_idx1
            while id_idx2 == id_idx1:
                id_idx2, img_idx2 = self._get_id_and_img_idx(np.random.randint(self.num_train))
            img2 = Image.open(self.img_dict[id_idx2]['images'][img_idx2]).convert('RGB')
            img2 = self.trans(img2)  # (C,H,W)
            ret_img2.append(img2)
            ret_id_idx2.append(id_idx2)
        return ret_img2, ret_id_idx2

    def _get_aug2_id_multi(self, id_aug1_list: list):
        ret_id_idx3, ret_img3 = [], []
        cnt = len(id_aug1_list)
        for i in range(cnt):
            id_idx3, img_idx3 = id_aug1_list[i], np.random.randint(self.images_per_id)
            img3 = Image.open(self.img_dict[id_idx3]['images'][img_idx3]).convert('RGB')
            img3 = self.trans(img3)  # (C,H,W)
            ret_img3.append(img3)
            ret_id_idx3.append(id_idx3)
        return ret_img3, ret_id_idx3

    @staticmethod
    def _mix_two_tensors(tensor_l: torch.Tensor, tensor_r: torch.Tensor, tensor_bg: torch.Tensor = None):
        c, h, w = tensor_l.shape
        assert tensor_l.shape == tensor_r.shape
        ret = torch.ones_like(tensor_l, device=tensor_l.device) * -1.  # (C,H,W)
        ret = tensor_bg if tensor_bg is not None else ret
        tensor_l = tensor_l.unsqueeze(0)
        tensor_r = tensor_r.unsqueeze(0)

        l_size_w = np.random.randint(h // 4, h // 4 * 3)
        l_size_h = int(l_size_w * np.random.uniform(0.8, 1.2))
        l_size_h = min(l_size_h, h)
        r_size_w = int((w - l_size_w) * np.random.uniform(0.9, 1.0))
        r_size_w = min(r_size_w, w - l_size_w)
        r_size_h = int(r_size_w * np.random.uniform(0.9, 1.1))
        r_size_h = min(r_size_h, h)

        tensor_l = F.interpolate(tensor_l, (l_size_h, l_size_w), mode='bilinear', align_corners=True)
        tensor_r = F.interpolate(tensor_r, (r_size_h, r_size_w), mode='bilinear', align_corners=True)

        l_pos_h = np.random.randint(h - l_size_h)
        l_pos_w = np.random.randint(w - l_size_w - r_size_w)
        l_pos_w = max(l_pos_w, 0)
        r_pos_h = np.random.randint(h - r_size_h)
        r_pos_w = np.random.randint(l_pos_w + l_size_w, w - r_size_w)
        r_pos_w = max(r_pos_w, 0)
        ret[:, l_pos_h: l_pos_h + l_size_h, l_pos_w: l_pos_w + l_size_w] = tensor_l[0]
        ret[:, r_pos_h: r_pos_h + r_size_h, r_pos_w: r_pos_w + r_size_w] = tensor_r[0]
        return ret  # (C,H,W)

    @staticmethod
    def _add_bg(tensor_img: torch.Tensor, tensor_bg: torch.Tensor = None, scale: list = None):
        c, h, w = tensor_img.shape
        # assert tensor_img.shape == tensor_bg.shape, '{} != {}'.format(tensor_img.shape, tensor_bg.shape)
        ret = torch.ones_like(tensor_img, device=tensor_img.device) * -1.  # (C,H,W)
        ret = tensor_bg if tensor_bg is not None else ret
        if scale is None:
            scale = [0.1, 1.0]
        rh = min(int(h * np.random.uniform(scale[0], scale[1])), h)
        rw = min(int(rh * np.random.uniform(0.9, 1.1)), w)
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), (rh, rw), mode='bilinear', align_corners=True)
        pos_h, pos_w = 0, 0
        if h > rh:
            pos_h = np.random.randint(h - rh)
        if w > rw:
            pos_w = np.random.randint(w - rw)

        ret[:, pos_h: pos_h + rh, pos_w: pos_w + rw] = tensor_img[0]
        return ret  # (C,H,W)


class FaceIdDatasetStyleGAN3(Dataset):
    def __init__(self,
                 pickle_path: str = '/gavin/datasets/stylegan/stylegan3-r-ffhq-1024x1024_ffhq.pickle',
                 num_ids: int = 10,
                 specific_ids: list = None,
                 image_size: int = 512,
                 repeats: int = 100,
                 flip_p: float = 0.5,
                 split: str = "train",
                 diff_cnt: int = 32,
                 **kwargs,
                 ):
        """ """
        super(FaceIdDatasetStyleGAN3, self).__init__()
        images_per_id = 1
        reg_ids = 0  # the num of regularization ids, default: 0, our method requires no regularization images
        reg_images_per_id = 1  # the min num of images per regularization id
        reg_repeats = 0  # repeat time of each regularization image
        asian = False

        if isinstance(specific_ids, str):
            import re
            if re.match(r'\d+-\d+', specific_ids) is not None:
                lo, hi = [int(x) for x in specific_ids.split('-')]
                specific_ids = list(np.arange(lo, hi))
            else:
                raise ValueError('Specific_ids not supported.')

        ''' full image list '''
        self.pickle_path = pickle_path
        self.num_ids = num_ids
        self.specific_ids = specific_ids
        self.images_per_id = images_per_id
        self.reg_ids = reg_ids
        self.reg_images_per_id = reg_images_per_id
        self.asian = asian
        self.diff_cnt = diff_cnt

        self.img_dict = {}  # {'id':str, 'images':[img]}
        self.img_list = []  # [img], all images
        self._load_from_pickle()

        self.repeats = repeats
        self.reg_repeats = reg_repeats
        self.split = split
        self.img_list = self.img_list[:num_ids * images_per_id] * repeats \
                        + self.img_list[num_ids * images_per_id:] * reg_repeats

        self.num_images = len(self.img_list)
        self.num_train = num_ids * images_per_id * repeats
        self.num_reg = reg_ids * self.reg_images_per_id * reg_repeats
        self._length = self.num_images

        ''' transform '''
        self.image_size = image_size
        self.trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.split == 'dev':
            self.trans = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=flip_p),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print('[FaceIdDataset] loaded from %s. (id*max_img=%d*%d, train%d+reg%d=total%d)' % (
            pickle_path,
            len(self.img_dict.keys()), len(self.img_dict[num_ids - 1]['images']),
            self.num_train, self.num_reg, self.num_images))

    def _load_from_pickle(self):
        """ """
        ''' 0. load all images '''
        with open(self.pickle_path, "rb") as handle:
            pickle_list = pickle.load(handle)
        pickle_dict = {}
        for img in pickle_list:
            id = os.path.basename(img).split('.')[0]
            if id not in pickle_dict:
                pickle_dict[id] = [img]
            else:
                pickle_dict[id] += [img]

        ''' 1. load train '''
        walk_id = 0
        use_id = 0
        train_id_set = set()
        for id, images in pickle_dict.items():
            if len(self.img_dict.keys()) >= self.num_ids:
                break  # enough
            if len(images) >= self.images_per_id:
                if self.specific_ids is not None and walk_id not in self.specific_ids:
                    walk_id += 1
                    continue
                self.img_dict[use_id] = {'id': id, 'images': images[:self.images_per_id]}
                self.img_list += images[:self.images_per_id]
                walk_id += 1
                use_id += 1
                train_id_set.add(id)
            if id == list(pickle_dict.keys())[-1] and use_id < self.num_ids:
                raise ValueError('Reach last. Not enough images for num_ids=%d, only %d.' % (
                    self.num_ids, len(train_id_set)))

        ''' 2. load reg '''
        reg_cnt = 0
        for id, images in pickle_dict.items():
            if len(self.img_dict.keys()) >= self.num_ids + self.reg_ids:
                break  # enough
            if id in train_id_set:
                continue  # Repeat with train
            if len(images) >= self.reg_images_per_id:
                self.img_dict[use_id] = {'id': id, 'images': [images[0]]}
                self.img_list += [images[0]]
                use_id += 1
                per_img_token_list.append('face')
                reg_cnt += 1
            if id == list(pickle_dict.keys())[-1]:
                raise ValueError('Reach last. Not enough images for reg_ids=%d, only %d.' % (
                    self.reg_ids, reg_cnt))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        ''' image '''
        id_idx, img_idx = self._get_id_and_img_idx(i)
        img = Image.open(self.img_dict[id_idx]['images'][img_idx]).convert('RGB')
        # equals to 'self.img_list[i]'
        img = self.trans(img)  # (C,H,W)

        ''' diff id '''
        img2_list, id_idx2_list = self._get_diff_id_multi(id_idx, img_idx, cnt=self.diff_cnt)

        ''' aug2 id '''
        img3_list, id_idx3_list = self._get_aug2_id_multi([id_idx] + id_idx2_list)

        dual_img = False if np.random.randint(10) < 50 else True
        if dual_img:  # (1,diff,1+diff)
            example["image_ori"] = {'faces': torch.cat([img.permute(1, 2, 0)]
                                                       + [img2.permute(1, 2, 0) for img2 in img2_list]
                                                       + [img3.permute(1, 2, 0) for img3 in img3_list],
                                                       dim=-1),  # (N,H,W,(1+diff+1+diff)C)
                                    'ids': torch.tensor([id_idx] + id_idx2_list + id_idx3_list),  # (N,1+diff+1+diff)
                                    'num_ids': 2}
            img = self._mix_two_tensors(img, img2_list[0], None)
        else:  # (1,diff,1+diff)
            example["image_ori"] = {'faces': torch.cat([img.permute(1, 2, 0)]
                                                       + [img2.permute(1, 2, 0) for img2 in img2_list]
                                                       + [img3.permute(1, 2, 0) for img3 in img3_list],
                                                       dim=-1),
                                    'ids': torch.tensor([id_idx] + id_idx2_list + id_idx3_list),
                                    'num_ids': 1}
            img = self._add_bg(img, None) if not self.split == 'dev' else img

        img = img.permute(1, 2, 0)  # in [-1,1], from (C,H,W) to (H,W,C)
        example["image"] = img
        example["id_idx"] = id_idx  # not used
        example["img_idx"] = img_idx  # not used

        ''' text '''
        placeholder_string = per_img_token_list[0]
        text = random.choice(imagenet_templates_small).format('face of %s person' % placeholder_string)
        if dual_img:
            str1, str2 = per_img_token_list[0], per_img_token_list[1]
            text = random.choice(imagenet_dual_templates_small).format('face of %s person' % str1,
                                                                       'face of %s person' % str2)
            text = '{}, left is a face of {} person, right is another face of {} person'.format(text, str1, str2)
        example["caption"] = text
        return example

    def _get_id_and_img_idx(self, i):
        if i < self.num_train:
            i %= (self.num_ids * self.images_per_id)
            id_idx = i // self.images_per_id
            img_idx = i % self.images_per_id
        else:
            i -= self.num_train
            i %= (self.reg_ids * 1)
            id_idx = i // self.reg_images_per_id + self.num_ids
            img_idx = i % self.reg_images_per_id
        return id_idx, img_idx

    def _get_diff_id_multi(self, id_idx1, img_idx1, cnt: int = 16):
        assert cnt < self.num_ids
        ret_id_idx2, ret_img2 = [], []
        for _ in range(cnt):
            id_idx2, img_idx2 = id_idx1, img_idx1
            while id_idx2 == id_idx1:
                id_idx2, img_idx2 = self._get_id_and_img_idx(np.random.randint(self.num_train))
            img2 = Image.open(self.img_dict[id_idx2]['images'][img_idx2]).convert('RGB')
            img2 = self.trans(img2)  # (C,H,W)
            ret_img2.append(img2)
            ret_id_idx2.append(id_idx2)
        return ret_img2, ret_id_idx2

    def _get_aug2_id_multi(self, id_aug1_list: list):
        ret_id_idx3, ret_img3 = [], []
        cnt = len(id_aug1_list)
        for i in range(cnt):
            id_idx3, img_idx3 = id_aug1_list[i], np.random.randint(self.images_per_id)
            img3 = Image.open(self.img_dict[id_idx3]['images'][img_idx3]).convert('RGB')
            img3 = self.trans(img3)  # (C,H,W)
            ret_img3.append(img3)
            ret_id_idx3.append(id_idx3)
        return ret_img3, ret_id_idx3

    @staticmethod
    def _mix_two_tensors(tensor_l: torch.Tensor, tensor_r: torch.Tensor, tensor_bg: torch.Tensor = None):
        c, h, w = tensor_l.shape
        assert tensor_l.shape == tensor_r.shape
        ret = torch.ones_like(tensor_l, device=tensor_l.device) * -1.  # (C,H,W)
        ret = tensor_bg if tensor_bg is not None else ret
        tensor_l = tensor_l.unsqueeze(0)
        tensor_r = tensor_r.unsqueeze(0)

        l_size_w = np.random.randint(h // 4, h // 4 * 3)
        l_size_h = int(l_size_w * np.random.uniform(0.8, 1.2))
        l_size_h = min(l_size_h, h)
        r_size_w = int((w - l_size_w) * np.random.uniform(0.9, 1.0))
        r_size_w = min(r_size_w, w - l_size_w)
        r_size_h = int(r_size_w * np.random.uniform(0.9, 1.1))
        r_size_h = min(r_size_h, h)

        tensor_l = F.interpolate(tensor_l, (l_size_h, l_size_w), mode='bilinear', align_corners=True)
        tensor_r = F.interpolate(tensor_r, (r_size_h, r_size_w), mode='bilinear', align_corners=True)

        l_pos_h = np.random.randint(h - l_size_h)
        l_pos_w = np.random.randint(w - l_size_w - r_size_w)
        l_pos_w = max(l_pos_w, 0)
        r_pos_h = np.random.randint(h - r_size_h)
        r_pos_w = np.random.randint(l_pos_w + l_size_w, w - r_size_w)
        r_pos_w = max(r_pos_w, 0)
        ret[:, l_pos_h: l_pos_h + l_size_h, l_pos_w: l_pos_w + l_size_w] = tensor_l[0]
        ret[:, r_pos_h: r_pos_h + r_size_h, r_pos_w: r_pos_w + r_size_w] = tensor_r[0]
        return ret  # (C,H,W)

    @staticmethod
    def _add_bg(tensor_img: torch.Tensor, tensor_bg: torch.Tensor = None, scale: list = None):
        c, h, w = tensor_img.shape
        # assert tensor_img.shape == tensor_bg.shape, '{} != {}'.format(tensor_img.shape, tensor_bg.shape)
        ret = torch.ones_like(tensor_img, device=tensor_img.device) * -1.  # (C,H,W)
        ret = tensor_bg if tensor_bg is not None else ret
        if scale is None:
            scale = [0.1, 1.0]
        rh = min(int(h * np.random.uniform(scale[0], scale[1])), h)
        rw = min(int(rh * np.random.uniform(0.9, 1.1)), w)
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), (rh, rw), mode='bilinear', align_corners=True)
        pos_h, pos_w = 0, 0
        if h > rh:
            pos_h = np.random.randint(h - rh)
        if w > rw:
            pos_w = np.random.randint(w - rw)

        ret[:, pos_h: pos_h + rh, pos_w: pos_w + rw] = tensor_img[0]
        return ret  # (C,H,W)


class FaceIdDatasetE4T(FaceIdDatasetStyleGAN3):
    def __init__(self,
                 pickle_path: str = '/gavin/datasets/aigc_id/dataset_e4t/ori_ffhq.pickle',
                 **kwargs,
                 ):
        """ """
        super(FaceIdDatasetE4T, self).__init__(pickle_path, **kwargs)


class FaceIdDatasetNobody(FaceIdDatasetStyleGAN3):
    def __init__(self,
                 pickle_path: str = '/gavin/datasets/aigc_id/dataset_nobody/ori_ffhq.pickle',
                 **kwargs,
                 ):
        """ """
        super(FaceIdDatasetNobody, self).__init__(pickle_path, **kwargs)


class FaceIdDatasetOneShot(FaceIdDatasetStyleGAN3):
    def __init__(self,
                 pickle_path: str,
                 **kwargs,
                 ):
        """ """
        super(FaceIdDatasetOneShot, self).__init__(pickle_path, **kwargs)


def tensor_to_arr(tensor):
    return ((tensor + 1.) * 127.5).cpu().numpy().astype(np.uint8)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    snapshot_folder = 'snapshot'
    os.system('rm -r %s' % snapshot_folder)
    os.makedirs(snapshot_folder, exist_ok=True)

    # bg_dataset = BackgroundDataset(split='val')
    # bg_loader = DataLoader(bg_dataset, batch_size=1, shuffle=False)
    # # tot_len = len(bg_dataset)
    # tot_len = 50
    # for k, sample in tqdm(enumerate(bg_loader)):
    #     index = k
    #     # index = np.random.randint(len(fid_dataset))
    #     img = sample
    #
    #     img = PIL.Image.fromarray(tensor_to_arr(img[0].permute(1, 2, 0)))
    #     img.save(os.path.join(snapshot_folder, 'bg_{:05d}.jpg'.format(k)))
    #
    #     if k >= tot_len:
    #         exit()

    # fid_dataset = FaceIdDatasetVGG("/gavin/datasets/original/image_512_quality.pickle",
    #                                split='dev',
    #                                num_ids=10,
    #                                specific_ids=[0, 1, 5, 6, 9,
    #                                              15, 28, 35, 62, 75],
    #                                images_per_id=10,
    #                                repeats=1,
    #                                reg_ids=1000,
    #                                reg_repeats=0,
    #                                asian=True,
    #                                diff_cnt=0,)
    # fid_dataset = FaceIdDatasetStyleGAN3(split='dev',
    #                                      num_ids=10,
    #                                      specific_ids='60-70',
    #                                      images_per_id=10,
    #                                      repeats=1,
    #                                      reg_ids=1000,
    #                                      reg_repeats=0,
    #                                      asian=True,
    #                                      diff_cnt=0, )
    # fid_dataset = FaceIdDatasetE4T(split='dev',
    #                                num_ids=1,
    #                                specific_ids=[6],
    #                                images_per_id=1,
    #                                repeats=1,
    #                                reg_ids=1000,
    #                                reg_repeats=0,
    #                                asian=True,
    #                                diff_cnt=0, )
    fid_dataset = FaceIdDatasetOneShot(
        pickle_path="/gavin/datasets/aigc_id/dataset_myself/ffhq.pickle",
        split='train',
        num_ids=1,
        specific_ids=[0],
        repeats=10,
        diff_cnt=0,
    )
    snapshot_loader = DataLoader(fid_dataset, batch_size=1, shuffle=False)
    tot_len = len(fid_dataset)
    # tot_len = 50
    for k, sample in tqdm(enumerate(snapshot_loader)):
        index = k
        # index = np.random.randint(len(fid_dataset))
        img = sample["image"]
        text = sample["caption"]
        id_idx = sample["id_idx"]
        img_idx = sample["img_idx"]
        image_ori = sample["image_ori"]
        # print(image_ori["faces"].shape, image_ori["ids"].shape)
        print(id_idx)

        img = PIL.Image.fromarray(tensor_to_arr(img[0]))
        img.save(os.path.join(snapshot_folder,
                              '{:05d}_id{}_#{}.jpg'.format(
                                  k, int(id_idx[0]), int(img_idx[0]), text[0])))

        if k >= tot_len:
            exit()
