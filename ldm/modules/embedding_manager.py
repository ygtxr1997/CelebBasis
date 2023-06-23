import torch
from torch import nn
from einops import rearrange
import numpy as np
from typing import List

from ldm.data.personalized import per_img_token_list
from ldm.modules.id_embedding.contrastive_loss import ContrastiveLoss
from ldm.modules.id_embedding.helpers import get_rep_pos, shift_tensor_dim0
from ldm.modules.id_embedding.meta_net import MetaIdNet
from transformers import CLIPTokenizer
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(
                    torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
            self,
            tokenized_text,
            embedded_text,
            face_image=None,
            img_ori=None,
            celeb_embeddings=None,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)

            if self.max_vectors_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                # print('token', placeholder_token, placeholder_token.shape)  # [265]
                # print('embedding', placeholder_embedding, placeholder_embedding.shape)  # (1,768)
                # print('tokenized_text', tokenized_text)  # [[x,x,x,265,x,x],[y,y,y,265,y,y]]
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding
                # print('placeholder_idx', placeholder_idx)  # [[0,1], [6,6]]
                # print('embedded_text after', embedded_text, embedded_text.shape)  # (2,77,768)
            else: # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def trainable_parameters(self):
        return []

    def embedding_to_coarse_loss(self):
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss

    def embedding_neg_loss(self):
        return 0.


class EmbeddingManagerId(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            max_ids: int = 10,
            num_embeds_per_token=1,
            momentum: float = 0.9,
            meta_mlp_depth: int = 2,
            loss_type: str = None,
            meta_inner_dim: int = 512,
            meta_heads: int = 1,
            use_rm_mlp: bool = False,
            test_mode: str = 'coefficient',  # coefficient/embedding/image/all
            save_fp16: bool = True,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        self.placeholder_strings = placeholder_strings
        self.max_ids = max_ids

        self.num_es = num_embeds_per_token
        self.meta_heads = meta_heads
        self.use_rm_mlp = use_rm_mlp

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        self.id_embeddings = [torch.zeros(
            num_embeds_per_token, token_dim
        )] * self.max_ids  # max_ids*(es,768)
        self.id_coefficients = [torch.randn(
            num_embeds_per_token, meta_heads, meta_inner_dim
        )] * self.max_ids  # max_ids*(es,head,inner_dim)
        self.celeb_embeddings = None  # (es,1+inner_dim,768)

        ''' 1. Placeholder mapping dicts '''
        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)
            self.string_to_token_dict[placeholder_string] = token  # fixed

        ''' 2. Identity dictionary '''
        for idx in range(self.max_ids):

            if initializer_words:
                init_word_token = get_token_for_string(initializer_words[0])
                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())
                token_params = torch.nn.Parameter(
                    init_word_embedding.unsqueeze(0).repeat(
                        self.num_es * self.meta_heads, 1), requires_grad=False)  # no backward
            else:
                token_params = torch.nn.Parameter(
                    torch.rand(size=(self.num_es * self.meta_heads, token_dim), requires_grad=False))

            self.id_embeddings[idx] = token_params  # each is (es*h,768)

        ''' id embedding '''
        self.meta_id_net = MetaIdNet(use_expert=False,
                                     mlp_depth=meta_mlp_depth,
                                     use_header=False,
                                     inner_dim=meta_inner_dim,
                                     meta_dim=token_dim,
                                     use_celebs=True,
                                     num_embeds_per_token=self.num_es,
                                     heads=self.meta_heads,
                                     use_rm_mlp=self.use_rm_mlp,
                                     # vis_mean=True,  # only for ablation study, make sure be False
                                     # vis_mean_params=(0., 0.1),  # only for ablation study,
                                     )
        self.momentum = momentum
        self.id_neg_loss = 0.
        self.moved_to_device = False
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.loss_type = loss_type
        if loss_type == 'contra':
            self.contra_loss = ContrastiveLoss(meta_dim=token_dim)
        self.test_mode = test_mode
        assert self.test_mode in ['coefficient', 'embedding', 'image', ]
        self.save_fp16 = save_fp16

    def forward(
            self,
            tokenized_text,  # (2,77)
            embedded_text,  # (2,77,768)
            face_image=None,  # not used
            img_ori=None,  # {'faces', 'ids', 'num_ids'}
            celeb_embeddings=None,  # (es,1+inner_dim,768), es:num_embeds_per_token
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        self.celeb_embeddings = celeb_embeddings.to(device)

        if img_ori is not None:
            faces = img_ori["faces"]  # (N,H,W,(1+diff+1+diff)C), id_cnt=1+diff+1+diff
            ids = img_ori["ids"]  # (N,(1+diff+1+diff))
            num_ids = img_ori["num_ids"]  # (N,)
            _, id_cnt = ids.shape
            if self.training or (faces is not None and self.test_mode == 'image'):
                meta, cls, cef = self.meta_id_net.forward_multi_faces(faces, ids, celeb_embeddings)
                # meta:id_cnt*(N,es,768), cef:id_cnt*(N,es,h,inner_dim)
                meta1 = meta[0]  # x:(N,es*h,768), the main id
                meta2 = meta[1]  # one of diff ids
                meta3 = meta[id_cnt // 2]  # if training: 2nd aug of x, for calculating 'contra' loss;
                cef1 = cef[0]  # cef1:(N,es,h,inner_dim)
                cef2 = cef[1]
                cef3 = cef[1]  # in training, the max #id is 2
                self._embedding_to_device(device)
                self._calc_id_neg_loss(meta, cls, ids, cef, device)
                if torch.isnan(meta[0][0]).sum() >= 1:
                    print('[Warning NAN detected][meta1]', meta[0][0].mean(), meta[0][0].min(), meta[0][0].max(),
                          meta[0].shape,
                          ids, num_ids,
                          'loss=', self.id_neg_loss)
            else:  # for testing
                meta1 = torch.randn(b)
                meta2 = torch.randn(b)
                meta3 = torch.randn(b)
                cef1 = torch.zeros(b)
                cef2 = torch.zeros(b)
                cef3 = torch.zeros(b)
                print(f'[Embedding Manager] test_mode: {self.test_mode}')

            ''' momentum update and swap embedded_text '''
            for b_idx in range(b):
                if num_ids[b_idx] == 2:  # two persons in an image
                    one_memo_l = self._momentum_update(meta1[b_idx], cef1[b_idx], ids[b_idx][0])  # refer to GroupNorm?
                    one_memo_r = self._momentum_update(meta2[b_idx], cef2[b_idx], ids[b_idx][1])
                    # print('[memo_l]', one_memo_l.mean(), one_memo_l.min(), one_memo_l.max())

                    placeholder_token_l = self.string_to_token_dict[self.placeholder_strings[0]]
                    placeholder_token_r = self.string_to_token_dict[self.placeholder_strings[1]]
                    placeholder_pos_lr = get_rep_pos(tokenized_text[b_idx],
                                                     [placeholder_token_l, placeholder_token_r])
                    # print('[tokenized]:', tokenized_text[b_idx])
                    # print('[tok_l, tok_r, pos_lr]:', placeholder_token_l, placeholder_token_r, placeholder_pos_lr)

                    embedded_text[b_idx], placeholder_final_pos_lr = shift_tensor_dim0(embedded_text[b_idx],
                                                                                       placeholder_pos_lr,
                                                                                       self.num_es * self.meta_heads)
                    # print('[shifted, final_pos_lr]:', embedded_text[b_idx].max(dim=-1)[0], placeholder_final_pos_lr)
                    placeholder_final_pos_l = placeholder_final_pos_lr[0]
                    placeholder_final_pos_r = placeholder_final_pos_lr[1]
                    for one_pos in placeholder_final_pos_l:
                        embedded_text[b_idx][one_pos] = one_memo_l.to(device)
                    for one_pos in placeholder_final_pos_r:
                        embedded_text[b_idx][one_pos] = one_memo_r.to(device)
                    # print('[replaced]:', embedded_text[b_idx].max(dim=-1)[0])

                elif num_ids[b_idx] == 1:  # one person in an image
                    one_memo = self._momentum_update(meta1[b_idx], cef1[b_idx], ids[b_idx][0])  # (es,768)

                    placeholder_token = self.string_to_token_dict[self.placeholder_strings[0]]
                    placeholder_pos = get_rep_pos(tokenized_text[b_idx],
                                                  [placeholder_token])
                    # print('[tokenized]:', tokenized_text[b_idx])
                    # print('[tok, pos]:', placeholder_token, placeholder_pos)
                    embedded_text[b_idx], placeholder_final_pos = shift_tensor_dim0(embedded_text[b_idx],
                                                                                    placeholder_pos,
                                                                                    self.num_es * self.meta_heads)
                    # print('[shifted, final_pos]:', embedded_text[b_idx].max(dim=-1)[0], placeholder_final_pos)
                    for one_pos in placeholder_final_pos[0]:
                        embedded_text[b_idx][one_pos] = one_memo.to(device)
                    # print('[replaced]:', embedded_text[b_idx].max(dim=-1)[0])

                elif num_ids[b_idx] == 3:  # three persons in an image
                    one_memo_l = self._momentum_update(meta1[b_idx], cef1[b_idx], ids[b_idx][0])  # refer to GroupNorm?
                    one_memo_r = self._momentum_update(meta2[b_idx], cef2[b_idx], ids[b_idx][1])
                    one_memo_3 = self._momentum_update(meta3[b_idx], cef3[b_idx], ids[b_idx][2])
                    # print('[memo_l]', one_memo_l.mean(), one_memo_l.min(), one_memo_l.max())

                    placeholder_token_l = self.string_to_token_dict[self.placeholder_strings[0]]
                    placeholder_token_r = self.string_to_token_dict[self.placeholder_strings[1]]
                    placeholder_token_3 = self.string_to_token_dict[self.placeholder_strings[2]]
                    placeholder_pos_lr = get_rep_pos(tokenized_text[b_idx],
                                                     [placeholder_token_l,
                                                      placeholder_token_r,
                                                      placeholder_token_3])
                    # print('[tokenized]:', tokenized_text[b_idx])
                    # print('[tok_l, tok_r, pos_lr]:', placeholder_token_l, placeholder_token_r, placeholder_pos_lr)

                    embedded_text[b_idx], placeholder_final_pos_lr = shift_tensor_dim0(embedded_text[b_idx],
                                                                                       placeholder_pos_lr,
                                                                                       self.num_es * self.meta_heads)
                    # print('[shifted, final_pos_lr]:', embedded_text[b_idx].max(dim=-1)[0], placeholder_final_pos_lr)
                    placeholder_final_pos_l = placeholder_final_pos_lr[0]
                    placeholder_final_pos_r = placeholder_final_pos_lr[1]
                    placeholder_final_pos_3 = placeholder_final_pos_lr[2]
                    for one_pos in placeholder_final_pos_l:
                        embedded_text[b_idx][one_pos] = one_memo_l.to(device)
                    for one_pos in placeholder_final_pos_r:
                        embedded_text[b_idx][one_pos] = one_memo_r.to(device)
                    for one_pos in placeholder_final_pos_3:
                        embedded_text[b_idx][one_pos] = one_memo_3.to(device)
                    # print('[replaced]:', embedded_text[b_idx].max(dim=-1)[0])

        return embedded_text

    def save(self, ckpt_path):
        save_dict = {}
        if self.save_fp16:  # 2Bytes/elem
            if self.test_mode == 'coefficient':
                save_dict["id_coefficients"] = [x.half() for x in self.id_coefficients]
            if self.test_mode == 'embedding':
                save_dict["id_embeddings"] = [x.half() for x in self.id_embeddings]
        else:  # 4Bytes/elem
            if self.test_mode == 'coefficient':
                save_dict["id_coefficients"] = self.id_coefficients
            if self.test_mode == 'embedding':
                save_dict["id_embeddings"] = self.id_embeddings
        if self.test_mode == 'image':
            save_dict["meta_id_net"] = self.meta_id_net.trainable_state_dict(verbose=True)
        torch.save(save_dict, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.id_coefficients = ckpt.get("id_coefficients")
        self.id_embeddings = ckpt.get("id_embeddings")

        if self.id_coefficients is not None:
            self.id_coefficients = [x.float() for x in self.id_coefficients]
        if self.id_embeddings is not None:
            self.id_embeddings = [x.float() for x in self.id_embeddings]

        if ckpt.get("meta_id_net") is not None:
            self.meta_id_net.load_trainable_state_dict(ckpt["meta_id_net"], verbose=True)
            self.meta_id_net.eval()
        print('[Embedding Manager] weights loaded.')

    def embedding_parameters(self):
        # return self.string_to_param_dict.parameters()
        return []  # to be compatible with vanilla Textual Inversion

    def trainable_parameters(self):
        trainable_list = []
        trainable_list.extend(list(self.meta_id_net.parameters()))
        if self.loss_type == 'contra':
            trainable_list.extend(list(self.contra_loss.parameters()))
        return trainable_list

    def embedding_to_coarse_loss(self):  # (from TextualInversion) not used
        loss = 0.
        return loss

    def _embedding_to_device(self, device):
        if self.moved_to_device:
            return
        for idx in range(len(self.id_embeddings)):
            self.id_embeddings[idx] = self.id_embeddings[idx].to(device)
        for idx in range(len(self.id_coefficients)):
            self.id_coefficients[idx] = self.id_coefficients[idx].to(device)
        self.moved_to_device = True

    def _momentum_update(self, one_pred_embedding: torch.Tensor,
                         one_pred_coefficient: torch.Tensor,
                         id_idx: int):
        """
        :param one_pred_embedding: (es*h,768)
        :param one_pred_coefficient: (es,h,inner_dim)
        :param id_idx: int
        """
        ''' 1. reg ids are out of range '''

        ''' 2. inner ids (testing) '''
        if not self.training:
            if self.test_mode == 'coefficient':
                if self.celeb_embeddings is None or self.id_coefficients is None:
                    print('[Warning] celeb_embeddings is None or id_coefficients is None.')
                    return one_pred_embedding.float()
                x = self.id_coefficients[id_idx]  # x:(es,h,inner_dim)
                self.celeb_embeddings = self.celeb_embeddings.to(x.device)
                c_mean, pca_base = self.celeb_embeddings[:, 0], self.celeb_embeddings[:, 1:]
                # mean:(es,768), pca_base:(es,inner_dim,768)
                c_mean = c_mean.unsqueeze(1)  # mean:(es,1,768)
                z = torch.einsum('e h k, e k c -> e h c', x, pca_base) + c_mean  # (es,h,768)
                z = rearrange(z, 'e h c -> (e h) c').contiguous()  # (num_es*heads,768)
                return z.float()
            elif self.test_mode == 'embedding':
                return self.id_embeddings[id_idx].float()
            elif self.test_mode == 'image':
                return one_pred_embedding.float()
            else:
                return one_pred_embedding.float()

        ''' 2. inner ids (training) '''
        if id_idx < len(self.id_embeddings):
            m = self.momentum
            self.id_embeddings[id_idx] = m * self.id_embeddings[id_idx] + (1 - m) * one_pred_embedding
            # self.id_embeddings[id_idx] = one_pred_embedding  # no momentum update
            self.id_coefficients[id_idx] = m * self.id_coefficients[id_idx] + (1 - m) * one_pred_coefficient
            # self.id_coefficients[id_idx] = one_pred_coefficient  # no momentum update
        return one_pred_embedding

    def _calc_id_neg_loss(self, meta, cls, ids, cef, device):
        if meta is None:
            return

        meta1 = meta[0]
        meta2s = meta[1:-1]
        meta3 = meta[-1]

        loss_cosine = 0.
        if self.loss_type == 'cosine':
            loss_cosine = (1 - torch.cosine_similarity(meta1, meta3))  # same
            for meta2 in meta2s:
                loss_cosine += torch.cosine_similarity(meta1, meta2)  # diff
            loss_cosine = loss_cosine.mean()

        loss_cls = 0.
        if cls is not None:
            _, id_cnt = ids.shape
            pred = torch.cat(cls, dim=0)  # (id_cnt*B,cls)
            label = torch.flatten(ids)  # (id_cnt*B,)
            loss_cls = self.cls_criterion(pred, label)

        loss_reg = 0.
        if cef is not None:
            cef = torch.cat(cef, dim=0)  # (id_cnt*B,meta_inner_dim)
        if self.loss_type == 'l1_reg':
            loss_reg = torch.norm(cef, dim=1, p=1).mean() * 1e-6
            print('[l1 reg loss]:', loss_reg)
        elif self.loss_type == 'l2_reg':
            loss_reg = torch.norm(cef, dim=1, p=2).mean() * 1e-6
            print('[l2 reg loss]:', loss_reg)

        loss_contra = 0.
        if self.loss_type == 'contra':
            loss_contra = self.contra_loss(meta) * 1e-2
            print('[contra loss]:', loss_contra)

        self.id_neg_loss = loss_cosine * 0 + loss_cls * 0 + loss_reg * 1 + loss_contra * 1

    def embedding_neg_loss(self):
        return self.id_neg_loss
