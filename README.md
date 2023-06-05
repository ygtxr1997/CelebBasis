# Inserting Anybody in Diffusion Models via Celeb Basis

[ArXiv](https://arxiv.org/abs/2306.00926) | [Project Page](https://celeb-basis.github.io) 

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


### How It Work
![Fig2](https://github.com/ygtxr1997/CelebBasis/assets/4397546/efe0eb13-0c74-45f0-9252-a49976dd228d)



First, we collect about 1,500 celebrity names as the initial collection. Then, we manually filter the initial one to m = 691 names, based on the synthesis quality of text-to-image diffusion model(stable-diffusion} with corresponding name prompt. Later, each filtered name is tokenized and encoded into a celeb embedding group. Finally, we conduct Principle Component Analysis to build a compact orthogonal basis.

![Fig4](https://github.com/ygtxr1997/CelebBasis/assets/4397546/fe70c970-f9d4-4255-bb76-0c6154778b4e)

We then personalize the model using input photo. During training~(left), we optimize the coefficients of the celeb basis with the help of a fixed face encoder. During inference~(right), we combine the learned personalized weights and shared celeb basis to generate images with the input identity.

More details can be found in our [project page](https://celeb-basis.github.io).

### TODO
- [ ] release code
- [ ] release celeb basis names
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
