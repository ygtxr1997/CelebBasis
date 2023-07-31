# Inserting Anybody in Diffusion Models via Celeb Basis

<a href='https://arxiv.org/abs/2306.00926'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://celeb-basis.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ygtxr1997/CelebBasis/blob/main/notebooks/CelebBasisColab.ipynb) &nbsp; 

<div>
<span class="author-block">
<a href="https://ygtxr1997.github.io/" target="_blank">Ge Yuan</a><sup>1,2</sup></span>,
<span class="author-block">
  <a href="http://vinthony.github.io/" target="_blank">Xiaodong Cun</a><sup>2</sup></span>,
<span class="author-block">
    <a href="https://yzhang2016.github.io" target="_blank">Yong Zhang</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=ym_t6QYAAAAJ&hl=zh-CN&oi=sra" target="_blank">Maomao Li</a><sup>2,*</sup>,
  </span>
<span class="author-block"><a href="https://chenyangqiqi.github.io/" target="_blank">Chenyang Qi</a><sup>3,2</sup></span>, <br>
  <span class="author-block">
    <a href="https://xinntao.github.io/" target="_blank">Xintao Wang</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ" target="_blank">Ying Shan</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CCUQi50AAAAJ" target="_blank">Huicheng Zheng</a><sup>1,*</sup>
  </span> (* Corresponding Authors)
  </div>

  
<div class="is-size-5 publication-authors">
                  <span class="author-block">
                  <sup>1</sup> Sun Yat-sen University &nbsp;&nbsp;&nbsp;
                  <sup>2</sup> Tencent AI Lab &nbsp;&nbsp;&nbsp;
                  <sup>3</sup> HKUST </span>
                </div>
<br>

**TL;DR: Intergrating a unique individual into the pre-trained diffusion model with:** 

✅ just <b>one</b> facial photograph &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
✅ only <b>1024</b> learnable parameters &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
✅ in <b>3</b> minutes tunning &nbsp;&nbsp;&nbsp;&nbsp;
</br>✅ Textural-Inversion compatibility &nbsp;&nbsp;&nbsp;&nbsp; ✅ Genearte and interact with other (new person) concepts </br>

![Fig1](https://github.com/ygtxr1997/CelebBasis/assets/4397546/f84a66aa-93ee-4b0a-9b81-5ab212019bb8)


### Updates
- **2023/06/23:** Code released!

### How It Work
![Fig2](https://github.com/ygtxr1997/CelebBasis/assets/4397546/efe0eb13-0c74-45f0-9252-a49976dd228d)



First, we collect about 1,500 celebrity names as the initial collection. Then, we manually filter the initial one to m = 691 names, based on the synthesis quality of text-to-image diffusion model(stable-diffusion} with corresponding name prompt. Later, each filtered name is tokenized and encoded into a celeb embedding group. Finally, we conduct Principle Component Analysis to build a compact orthogonal basis.

![Fig4](https://github.com/ygtxr1997/CelebBasis/assets/4397546/fe70c970-f9d4-4255-bb76-0c6154778b4e)

We then personalize the model using input photo. During training~(left), we optimize the coefficients of the celeb basis with the help of a fixed face encoder. During inference~(right), we combine the learned personalized weights and shared celeb basis to generate images with the input identity.

More details can be found in our [project page](https://celeb-basis.github.io).


### Setup

Our code mainly bases on [Textual Inversion](https://github.com/rinongal/textual_inversion).
We add some environment requirements for Face Alignment & Recognition to the original environment of [Textual Inversion](https://github.com/rinongal/textual_inversion).
To set up our environment, please run:

```shell
conda env create -f environment.yaml
conda activate sd
```

The pre-trained weights used in this repo include [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and 
[CosFace R100 trained on Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo).
You may copy these pre-trained weights to `./weights`, and the directory tree will be like:

```shell
CelebBasis/
  |-- weights/
      |--glint360k_cosface_r100_fp16_0.1/
          |-- backbone.pth (249MB)
      |--sd-v1-4-full-ema.ckpt (7.17GB)
```

We use [PIPNet](https://github.com/jhb86253817/PIPNet) to align and crop the face.
The PIPNet pre-trained weights can be downloaded from [this link](https://github.com/ygtxr1997/CelebBasis/issues/2#issuecomment-1607775140) (provided by @justindujardin)
or our [Baidu Yun Drive](https://pan.baidu.com/s/1Cgw0i723SyeLo5lbJu-b0Q) with extracting code: `ygss`.
Please copy `epoch59.pth` and `FaceBoxesV2.pth` to `CelebBasis/evaluation/face_align/PIPNet/weights/`.

### Usage

#### 0. Face Alignment

To make the Face Recognition model work as expected, 
given an image of a person, 
we first align and crop the face following [FFHQ-Dataset](https://github.com/NVlabs/ffhq-dataset).

Assuming your image folder is `/Your/Path/To/Images/ori/` and the output folder is `/Your/Path/To/Image/ffhq/`,
you may run the following command to align & crop images.

```shell
bash ./00_align_face.sh /Your/Path/To/Images/ori /Your/Path/To/Images/ffhq
```

Then, a pickle file named `ffhq.pickle` using **absolute path** will be generated under `/Your/Path/To/Images/`, 
which is used for training dataset setting later.
For example, we provide the original and cropped [StyleGAN generated faces](https://github.com/NVlabs/stylegan3-detector) 
in [Baiduyun Drive (code:ygss)](https://pan.baidu.com/s/1_W-tlBwY4S8t3_bPtPlJ5g), where:
- `stylegan3-r-ffhq-1024x1024` is the original images (`/Your/Path/To/Images/ori`)
- `stylegan3-r-ffhq-1024x1024_ffhq` is the cropped images (`/Your/Path/To/Image/ffhq/`)
- `stylegan3-r-ffhq-1024x1024_ffhq.pickle` is the pickle list file (`/Your/Path/To/Images/ffhq.pickle`)

We also provide some cropped faces in `./infer_images/dataset_stylegan3_10id/ffhq` as the example and reference.

#### 1. Personalization

The training config file is `./configs/stable-diffusion/aigc_id.yaml`.
The most important settings are listed as follows.

**Important Data Settings**
```yaml
data:
  params:
    batch_size: 2  # We use batch_size 2
    train:
      target: ldm.data.face_id.FaceIdDatasetOneShot  # or ldm.data.face_id.FaceIdDatasetStyleGAN3
      params:
        pickle_path: /Your/Path/To/Images/ffhq.pickle  # pickle file generated by Face Alignment, consistent with 'target'
        num_ids: 2  # how many IDs used for jointly training
        specific_ids: [1,2]  # you may specify the index of ID for training, e.g. [0,1,2,3,4,5,6,7,8,9], 0 means the first
    validation:
      target: ldm.data.face_id.FaceIdDatasetOneShot
      params:
        pickle_path: /Your/Path/To/Images/ffhq.pickle  # consistent with train.params.pickle_path
```

**Important Model Settings**
```yaml
model:
  params:
    personalization_config:
      target: ldm.modules.embedding_manager.EmbeddingManagerId
      params:
        max_ids: 10  # max joint learning #ids, should >= data.train.num_ids
        num_embeds_per_token: 2  # consistent with [cond_stage_config]
        meta_mlp_depth: 1  # single layer is ok
        meta_inner_dim: 512  # consistent with [n_components]
        test_mode: 'coefficient'  # coefficient/embedding/image/all
        momentum: 0.99  # momentum update the saved dictionary
        save_fp16: False  # save FP16, default is FP32

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
      params:
        use_celeb: True  # use celeb basis
        use_svd: True  # use SVD version of PCA
        rm_repeats: True  # removing repeated words can be better
        celeb_txt: "./infer_images/wiki_names_v2.txt"  # celebs, wiki_names_v1, wiki_names_v2.txt
        n_components: 512  # consistent with [meta_inner_dim]
        use_flatten: False  # flattening means dropping the word position information
        num_embeds_per_token: 2  # consistent with [personalization_config]
```

**Important Training Settings**
```yaml
lightning:
  modelcheckpoint:
    params:
      every_n_train_steps: 200  # 100x num of IDs
  callbacks:
    image_logger:
      params:
        batch_frequency: 600  # 300x num of IDs
  trainer:
    max_steps: 800  # 400x num of IDs
```

**Training**
```shell
bash ./01_start_train.sh ./weights/sd-v1-4-full-ema.ckpt
```

Consequently, a project folder named `traininYYYY-MM-DDTHH-MM-SS_celebbasis` is generated under `./logs`. 

#### 2. Generation

Edit the prompt file `./infer_images/example_prompt.txt`, where `sks` denotes the first identity 
and `ks` denotes the second identity.

Optionally, in `./02_start_test.sh`, you may modify the following var as you need:
```shell
step_list=(799)  # the step of trained '.pt' files, e.g. (99 199 299 399)
eval_id1_list=(0)  # the ID index of the 1st person, e.g. (0 1 2 3 4)
eval_id2_list=(1)  # the ID index of the 2nd person, e.g. (0 1 2 3 4)
```

**Testing**
```shell
bash ./02_start_test.sh "./weights/sd-v1-4-full-ema.ckpt" "./infer_images/example_prompt.txt" "traininYYYY-MM-DDTHH-MM-SS_celebbasis"
```

The generated images are under `./outputs/traininYYYY-MM-DDTHH-MM-SS_celebbasis`.

#### 3. (Optional) Extracting ID Coefficients

Optionally, you can extract the coefficients for each identity by running:

```shell
bash ./03_extract.sh "./weights/sd-v1-4-full-ema.ckpt" "traininYYYY-MM-DDTHH-MM-SS_celebbasis"
```

The extracted coefficients or embeddings are under `./weights/ti_id_embeddings/`.

### TODO
- [x] release code
- [x] release celeb basis names
- [ ] release google colab project
- [ ] release WebUI extension
- [ ] release automatic name filter
- [ ] finetuning with multiple persons 
- [ ] finetuning with LORA

### BibTex

```tex
@article{yuan2023celebbasis,
  title={Inserting Anybody in Diffusion Models via Celeb Basis},
  author={Yuan, Ge and Cun, Xiaodong and Zhang, Yong and Li, Maomao and Qi, Chenyang and Wang, Xintao and Shan, Ying and Zheng, Huicheng},
  journal={arXiv preprint arXiv:2306.00926},
  year={2023}
}
```
