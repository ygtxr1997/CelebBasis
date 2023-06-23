import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error

import clip
import torch
import torch.nn.functional as F
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler

from evaluation.base_class import EvaluatorBase
from evaluation.face_align.PIPNet.alignment.alignment import norm_crop
from evaluation.face_align.PIPNet.alignment.landmarks import get_5_from_98
from evaluation.face_align.PIPNet.lib.tools import get_lmk_model, demo_image
from ldm.modules.id_embedding.iresnet import iresnet100
from evaluation.face_align import cosface


class CLIPEvaluator(EvaluatorBase):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()

    def evaluate(self, gen_samples, src_images, target_text):
        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)
        return sim_samples_to_img, sim_samples_to_text


class LDMCLIPEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, ldm_model, src_images, target_text, n_samples=64, n_steps=50):
        
        sampler = DDIMSampler(ldm_model)

        samples_per_batch = 8
        n_batches         = n_samples // samples_per_batch

        # generate samples
        all_samples=list()
        with torch.no_grad():
            with ldm_model.ema_scope():                
                uc = ldm_model.get_learned_conditioning(samples_per_batch * [""])

                for batch in range(n_batches):
                    c = ldm_model.get_learned_conditioning(samples_per_batch * [target_text])
                    shape = [4, 256//8, 256//8]
                    samples_ddim, _ = sampler.sample(S=n_steps,
                                                    conditioning=c,
                                                    batch_size=samples_per_batch,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=5.0,
                                                    unconditional_conditioning=uc,
                                                    eta=0.0)

                    x_samples_ddim = ldm_model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)

                    all_samples.append(x_samples_ddim)
        
        all_samples = torch.cat(all_samples, axis=0)

        sim_samples_to_img  = self.img_to_img_similarity(src_images, all_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), all_samples)

        return sim_samples_to_img, sim_samples_to_text


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):

        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)

        return sim_samples_to_img, sim_samples_to_text


class IdentityEvaluator(object):
    def __init__(self,
                 device: torch.device,
                 align_mode: str = 'ffhq',
                 img_size: int = 512,
                 ):
        self.align_mode = align_mode
        self.img_size = img_size

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        self.trans_arr_to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
        ])
        print('[IdentityEvaluator] alignment model loaded')

        ''' face recognition '''
        self.id_model = None
        self._load_fr_net()
        self.trans_matrix = torch.tensor(
            [
                [
                    [1.07695457, -0.03625215, -1.56352194 / 512],
                    [0.03625215, 1.07695457, -5.32134629 / 512],
                ]
            ],
            requires_grad=False,
        ).float().cuda()
        print('[IdentityEvaluator] face recognition model loaded')

    @torch.no_grad()
    def start_calc(self, ori1: torch.Tensor, ori2: torch.Tensor):
        n1, c, h, w = ori1.shape
        n2, c, h, w = ori2.shape
        crops, num_has_face, num_no_face = self._check_lmk_box_for_tensor(
            torch.cat([ori1, ori2], dim=0))
        crop1 = crops[:n1]
        crop2 = crops[n1:]
        avg_cos_sim, avg_mse_dist, avg_l2_dist = self._img_to_img_id_sim(crop1, crop2)
        return {
            "cos_sim": avg_cos_sim,
            "mse_dist": avg_mse_dist,
            "l2_dist": avg_l2_dist,
            "num_has_face": num_has_face,
            "num_no_face": num_no_face
        }

    def _check_lmk_box_for_tensor(self, img_tensor: torch.Tensor):
        b, c, h, w = img_tensor.shape
        img_arr = ((img_tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        img_cropped_first = None
        img_cropped = []

        num_has_face, num_no_face = 0, 0
        for i in range(b):
            img = img_arr[i]
            cropped, success = self._check_lmk_box_for_one_image(img)
            if i == 0:
                img_cropped_first = cropped
            if success or i == 0:
                num_has_face += 1
                img_cropped.append(self.trans_arr_to_tensor(cropped).to(img_tensor.device))
            else:
                num_no_face += 1
        img_cropped = torch.stack(img_cropped, dim=0)
        return img_cropped, num_has_face, num_no_face

    def _check_lmk_box_for_one_image(self, img_arr: np.ndarray):
        full_img = img_arr.astype(np.uint8)
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, self.img_size, mode=self.align_mode, borderValue=0.0)
            return cropped_img, True
        else:
            return full_img, False

    def _img_to_img_id_sim(self, face1: torch.Tensor, face2: torch.Tensor):
        n1, c, h, w = face1.shape
        n2, c, h, w = face2.shape
        if n1 < 1 or n2 < 1:
            return 0, 0, 0
        faces = torch.cat([face1, face2], dim=0)

        ''' align to insightface '''
        M = self.trans_matrix.repeat(faces.size()[0], 1, 1)  # to (B,2,3)
        grid = F.affine_grid(M, size=faces.size(), align_corners=True)  # 得到grid 用于grid sample
        faces = F.grid_sample(faces, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine
        faces = F.interpolate(faces, size=112, mode="bilinear", align_corners=True)

        feats = self.id_model(faces)
        feats = F.normalize(feats, dim=-1, p=2)
        feat1 = feats[:n1]
        feat2 = feats[n1:]

        cos_sim = F.cosine_similarity(feat1[:, None, :], feat2[None, :, :], dim=-1)  # (n1,n2)
        mse_dist, l2_dist, dim = self._calc_mse_l2(feat1.repeat(n2, 1).cpu().numpy(),
                                                   feat2.cpu().numpy())
        return cos_sim.mean(), mse_dist, l2_dist

    @staticmethod
    def _calc_mse_l2(vec1: np.ndarray, vec2: np.ndarray):
        mse_dist = mean_squared_error(vec1, vec2)
        ''' L2 = sqrt(MSE * dim) '''
        dim = vec1.shape[-1]
        l2_dist = np.sqrt(mse_dist * dim) / 2
        return float(mse_dist), float(l2_dist), dim

    def _load_fr_net(self):
        # self.id_model = iresnet100()
        # id_path = '/gavin/code/FaceSwapping/modules/third_party/arcface/weights/' \
        #           'ms1mv3_arcface_r100_fp16/backbone.pth'
        self.id_model = cosface.net.sphere().cuda()
        id_path = '/gavin/code/FaceSwapping/modules/third_party/cosface/net_sphere20_data_vggface2_acc_9955.pth'

        weights = torch.load(id_path)
        self.id_model.load_state_dict(weights)
        for param in self.id_model.parameters():
            param.requires_grad = False
        self.id_model.eval()
        self.id_model = self.id_model.cuda()


class IdCLIPEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32',
                 align_mode: str = 'ffhq',
                 img_size: int = 512,
                 ) -> None:
        super().__init__(device, clip_model)
        self.id_evaluator = IdentityEvaluator(
            device, align_mode, img_size
        )
        print('[IdCLIPEvaluator] ID and CLIP evaluator loaded.')

    def evaluate(self, gen_samples: torch.Tensor,
                 src_images: torch.Tensor, target_text: str):

        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, gen_samples)
        id_result_dict = self.id_evaluator.start_calc(src_images, gen_samples)

        return sim_samples_to_img, sim_samples_to_text, id_result_dict
