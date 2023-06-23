import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _build_causal_attention_mask(bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text, embedding_manager=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True, embedding_manager=embedding_manager)
        return z

    def encode(self, text, **kwargs):
        # output of length 77
        return self(text, **kwargs)

class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 celeb_txt="/gavin/code/TextualInversion/infer_images/wiki_names.txt",
                 use_celeb=False,
                 use_svd=False,
                 n_components: int = 512,
                 rm_repeats: bool = True,
                 use_sample_reduce=False,
                 n_samples: int = 513,
                 use_flatten: bool = True,
                 num_embeds_per_token: int = 2,
                 ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

        def embedding_forward(
                self,
                input_ids = None,
                position_ids = None,
                inputs_embeds = None,
                embedding_manager = None,
                face_img = None,
                image_ori=None,
                only_embedding=False,
                celeb_embeddings=None,
            ) -> torch.Tensor:

                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

                two_celebs = False
                multi_celebs = False
                word_interpolation = False

                if two_celebs:
                    print('input_ids:', input_ids)
                    input_ids_2 = input_ids.clone()
                    if input_ids_2.ndim > 1:
                        input_ids_2[:, :] = 49407
                        input_ids_2[:, 0] = 49406
                        input_ids_2[0, 1] = 320     # a
                        input_ids_2[0, 2] = 1125    # photo
                        input_ids_2[0, 3] = 539     # of
                        input_ids_2[0, 4] = 20406   # Elon
                        input_ids_2[0, 5] = 19063   # Musk
                        input_ids_2[1, 1] = 4083    # Anne
                        input_ids_2[1, 2] = 31801   # Hathaway
                        input_ids_2[1, 3] = 22481   # Barack
                        input_ids_2[1, 4] = 4276    # Obama
                        input_ids_2[1, 5] = 3929    # Robert
                        input_ids_2[1, 6] = 29429   # Downey

                if word_interpolation:
                    print('input_ids:', input_ids)
                    input_ids_2 = input_ids.clone()
                    if input_ids_2.ndim > 1:
                        input_ids_2[:, :] = 49407
                        input_ids_2[:, 0] = 49406
                        input_ids_2[0, 1] = 320  # a
                        input_ids_2[0, 2] = 1125  # photo
                        input_ids_2[0, 3] = 539  # of
                        input_ids_2[0, 4] = 20406  # Elon
                        input_ids_2[0, 5] = 19063  # Musk
                        input_ids_2[0, 6] = 1674  # picture
                        input_ids_2[1, 1] = 4083  # Anne
                        input_ids_2[1, 2] = 31801  # Hathaway
                        input_ids_2[1, 3] = 22481  # Barack
                        input_ids_2[1, 4] = 4276  # Obama
                        input_ids_2[1, 5] = 3929  # Robert
                        input_ids_2[1, 6] = 29429  # Downey
                        input_ids_2[1, 19] = 2888
                        input_ids_2[1, 20] = 3333

                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]

                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)
                    if only_embedding:
                        return inputs_embeds

                    if two_celebs and input_ids_2.ndim > 1:
                        inputs_embeds_2 = self.token_embedding(input_ids_2)
                        alpha1 = 0.2  # Elon only 0.2
                        alpha2 = 0.6  # Elon musk 0.5

                        emb_elon = inputs_embeds_2[0, 4]
                        emb_musk = inputs_embeds_2[0, 5]
                        emb_anne = inputs_embeds_2[1, 1]
                        emb_hathaway = inputs_embeds_2[1, 2]
                        emb_barack = inputs_embeds_2[1, 3]
                        emb_obama = inputs_embeds_2[1, 4]
                        emb_robert = inputs_embeds_2[1, 5]
                        emb_downey = inputs_embeds_2[1, 6]

                        emb_id1 = [alpha1 * emb_elon + (1 - alpha1) * emb_anne, emb_musk]  # ata tre
                        emb_id2 = [alpha2 * emb_robert + (1 - alpha2) * emb_barack,
                                   alpha2 * emb_downey + (1 - alpha2) * emb_obama]  # sks ks

                        pos_id1_1 = torch.where(input_ids == 4236)  # ata
                        pos_id1_2 = torch.where(input_ids == 6033)  # tre
                        pos_id2_1 = torch.where(input_ids == 48136)  # sks
                        pos_id2_2 = torch.where(input_ids == 662)    # ks

                        # print(pos_id2_1, pos_id2_2)
                        inputs_embeds[pos_id1_1] = emb_id1[0]
                        inputs_embeds[pos_id1_2] = emb_id1[1]
                        inputs_embeds[pos_id2_1] = emb_id2[0]
                        inputs_embeds[pos_id2_2] = emb_id2[1]

                    if word_interpolation and input_ids_2.ndim > 1:
                        inputs_embeds_2 = self.token_embedding(input_ids_2)
                        alpha1 = 0.01  # Elon only 0.2
                        alpha2 = 0.6  # Elon musk 0.5

                        emb_photo = inputs_embeds_2[0, 2]
                        emb_picture = inputs_embeds_2[0, 6]
                        emb_other1 = inputs_embeds_2[1, 19]
                        emb_other2 = inputs_embeds_2[1, 20]

                        emb_word1 = [alpha1 * emb_other1 + (1 - alpha1) * emb_other2, ]  # ata

                        pos_id1_1 = torch.where(input_ids == 4236)  # ata
                        pos_id1_2 = torch.where(input_ids == 6033)  # tre
                        pos_id2_1 = torch.where(input_ids == 48136)  # sks
                        pos_id2_2 = torch.where(input_ids == 662)    # ks

                        # print(pos_id1_1, pos_id1_2)
                        inputs_embeds[pos_id1_1] = emb_word1[0]

                if embedding_manager is not None and not two_celebs and not multi_celebs and not word_interpolation:
                    # call EmbeddingManager.forward(tokens, embeddings)
                    inputs_embeds = embedding_manager(input_ids, inputs_embeds, face_img, image_ori,
                                                      celeb_embeddings)

                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                
                return embeddings      

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(self.transformer.text_model.embeddings)

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask = None,
            causal_attention_mask = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                coloutputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = coloutputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (coloutputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(self.transformer.text_model.encoder)


        def text_encoder_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
            face_img=None,
            image_ori=None,
            only_embedding=False,
            celeb_embeddings=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                            embedding_manager=embedding_manager,
                                            face_img=face_img,
                                            image_ori=image_ori,
                                            only_embedding=only_embedding,
                                            celeb_embeddings=celeb_embeddings,
                                            )
            if only_embedding:
                return hidden_states

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
            face_img = None,
            image_ori=None,
            only_embedding: bool = False,
            celeb_embeddings=None
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
                face_img=face_img,
                image_ori=image_ori,
                only_embedding=only_embedding,
                celeb_embeddings=celeb_embeddings,
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

        self.dev = False
        if self.dev:
            celeb_txt = "./infer_images/wiki_names.txt"
        self.celeb_txt = celeb_txt
        self.celeb_embeddings = None
        self.use_celeb = use_celeb
        self.use_svd = use_svd
        self.rm_repeats = rm_repeats
        self.use_sample_reduce = use_sample_reduce
        self.n_samples = n_samples
        self.use_flatten = use_flatten
        self.num_embeds_per_token = num_embeds_per_token
        if use_celeb:
            self._get_celeb_embeddings(n_components)
            self.n_components = n_components

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)        
        z = self.transformer(input_ids=tokens,
                             celeb_embeddings=self.celeb_embeddings,
                             **kwargs)

        return z

    def encode(self, text, **kwargs):
        return self(text, **kwargs)

    @torch.no_grad()
    def _get_celeb_embeddings(self, n_components: int = 0):
        with open(self.celeb_txt, "r") as f:
            celeb_names = f.read().splitlines()

        if self.rm_repeats:
            # remove repeated names
            celeb_set = set()
            for name in celeb_names:
                celeb_set.add(name)
        else:
            celeb_set = celeb_names

        celeb_list = list(celeb_set)  # set outputs are randomly ordered!!!
        celeb_list.sort()

        ''' get tokens and embeddings '''
        tokens_list = []
        embeddings_list = []
        for name in celeb_list:
            batch_encoding = self.tokenizer(name, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]  # (1,77)
            embeddings = self.transformer(input_ids=tokens, only_embedding=True)  # (1,77,768)

            tokens_list.append(tokens)
            embeddings_list.append(embeddings)
        all_tokens: torch.Tensor = torch.cat(tokens_list, dim=0)  # (M,77)
        all_embeddings: torch.Tensor = torch.cat(embeddings_list, dim=0)  # (M,77,768)

        ''' measure length '''
        if self.dev:
            import numpy as np
            with open('./infer_images/token_len.txt', 'w') as f:
                lines = []
                for m in range(len(celeb_list)):
                    name = celeb_list[m]
                    name_tok = all_tokens[m][torch.where(all_tokens[m] < 49406)]
                    token_num = name_tok.shape[0]
                    token_list = name_tok.cpu().numpy().astype(np.uint32).tolist()
                    lines.append('{:04d} {}: len={}, token={}\n'.format(m, name, token_num, token_list))
                f.writelines(lines)
            print('DEV mode. existing...')
            exit()

        flatten = self.use_flatten
        cols_embeddings = []
        flat_embeddings = []
        all_token_set = {torch.Tensor([49406]), torch.Tensor([49407])}
        if not flatten:
            for j in range(all_tokens.shape[1]):  # col
                col_token_set = {torch.Tensor([49406]), torch.Tensor([49407])}
                col_embeddings = []
                for i in range(all_tokens.shape[0]):  # row
                    tok = all_tokens[i, j]
                    if tok >= 49406:
                        continue
                    if self.rm_repeats and (tok in col_token_set):
                        continue
                    if flatten and self.rm_repeats and (tok in all_token_set):
                        continue  # if flatten=True, any repeated words are unacceptable
                    col_embeddings.append(all_embeddings[i, j].unsqueeze(0))
                    flat_embeddings.append(all_embeddings[i, j].unsqueeze(0))
                    col_token_set.add(tok)
                    all_token_set.add(tok)
                if len(col_embeddings) > 0:
                    cols_embeddings.append(col_embeddings)
        else:
            for i in range(all_tokens.shape[0]):  # row
                for j in range(all_tokens.shape[1]):  # col
                    tok = all_tokens[i, j]
                    if tok >= 49406:
                        continue
                    if flatten and self.rm_repeats and (tok in all_token_set):
                        continue  # if flatten=True, any repeated words are unacceptable
                    flat_embeddings.append(all_embeddings[i, j].unsqueeze(0))
                    all_token_set.add(tok)

        if flatten:
            cols_embeddings = [flat_embeddings]
        print('[celebs cols_embeddings len]:', [len(x) for x in cols_embeddings])

        use_token_cols = self.num_embeds_per_token
        self.celeb_embeddings = []
        for j in range(len(cols_embeddings)):
            if j >= use_token_cols:
                break
            col_embeddings = cols_embeddings[j]
            col_celeb_embeddings: torch.Tensor = torch.cat(col_embeddings, dim=0)  # (k,768)
            print('[(col_{}) celebs ori]:'.format(j), col_celeb_embeddings.shape, col_celeb_embeddings[0].mean(),
                  col_celeb_embeddings[0].min(), col_celeb_embeddings[0].max())

            ''' QR '''
            # celeb_embeddings = celeb_embeddings.t()
            # q, r = torch.qr(celeb_embeddings)  # bases, up-triangle

            ''' sklearn PCA validate '''
            # if True:
            #     import numpy as np
            #     from sklearn.decomposition import PCA
            #     x = celeb_embeddings.t().cpu().numpy()  # (m,768)->(768,m)
            #     pca = PCA(n_components=self.n_samples)
            #     res = pca.fit_transform(x)
            #     print('[celebs sklearn PCA]:', res.shape, res.mean(),
            #           res.min(), res.max())

            ''' reduce #samples '''
            if self.use_sample_reduce:
                celeb_embeddings = col_celeb_embeddings.t()  # (m,768)->(768,m)
                n, m = celeb_embeddings.shape  # (768,m)
                x = celeb_embeddings - celeb_embeddings.mean(dim=0, keepdims=True)
                u, s, v = torch.svd(x, some=False)  # u:(768,768), v:(m,m), v[:,0]=e0
                vr = v[:, :self.n_samples]  # (m,r), is pca_base
                celeb_embeddings = torch.matmul(celeb_embeddings, vr)  # (768,r)
                col_celeb_embeddings = celeb_embeddings.t()  # (r,768)
                print('[(col_{}) celebs reduced]:'.format(j), col_celeb_embeddings.shape, col_celeb_embeddings.mean(),
                      col_celeb_embeddings.min(), col_celeb_embeddings.max())

            ''' svd '''
            if self.use_svd:
                ''' op1. prime '''
                # u, s, v = torch.svd(celeb_embeddings, some=False)  # u:(513,513), v:(768,768), v[:,0]=e0
                # # print(u.shape, s.shape, v.shape)
                # # print(v[0] * v[1])
                # # print(v[:, 0] * v[:, 1])
                # print('[rank of celebs]:', torch.sum(s > 1e-10))
                # self.celeb_embeddings = v.t()

                ''' op2. 3dmm/pca-based svd '''
                m, n = col_celeb_embeddings.shape  # (646,768), assume inner_dim=512
                c_mean = col_celeb_embeddings.mean(dim=0, keepdims=True)  # (1,768)
                x = col_celeb_embeddings - c_mean
                # cov = 1. / m * torch.matmul(x.t(), x)  # (768,768)
                u, s, v = torch.svd(x, some=False)  # u:(646,646), v:(768,768), v[:,0]=e0
                vt = v.t()[:n_components]  # (n_components,768), is pca_base
                col_celeb_embeddings = torch.cat([c_mean, vt], dim=0)  # (1+n_components,768), 0:mean, others:pca_base
                print('[(col_{}) celebs c_mean]:'.format(j), c_mean.mean(), c_mean.min(), c_mean.max())
                print('[(col_{}) celebs pca_base]:'.format(j), vt.shape, vt[0].mean(),
                      vt[0].min(), vt[0].max(), vt[0].norm())

            col_celeb_embeddings.requires_grad_(False)
            col_celeb_embeddings = col_celeb_embeddings.to(self.device)
            print('[(col_{}) celebs embeddings loaded]:'.format(j), col_celeb_embeddings.shape,
                  '(device:{})'.format(self.device))  # (1+n_components,768)

            self.celeb_embeddings.append(col_celeb_embeddings.unsqueeze(0))

        if flatten:
            self.celeb_embeddings = self.celeb_embeddings * use_token_cols
        self.celeb_embeddings = torch.cat(self.celeb_embeddings, dim=0)
        print('[all celebs embeddings loaded] shape =', self.celeb_embeddings.shape,
              '(device:{})'.format(self.device),
              'flatten={}'.format(flatten))  # (num_embeds_per_token,1+n_components,768), e.g. (2,513,768)

    @torch.no_grad()
    def save_celeb_embeddings(self, pth_path: str):
        assert isinstance(self.celeb_embeddings, torch.Tensor), "No celeb_embeddings!"
        celeb_embeddings = self.celeb_embeddings.cpu()
        torch.save(celeb_embeddings, pth_path)
        print(f"[FrozenCLIPEmbedder] celeb basis saved to {pth_path}.")


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)