# TAB (Tuning with Auxiliary Branch)
Code and pretrained models for ACL 2023 main conference short paper: "[Modality Adaption or Regularization? A Case Study on End-to-End Speech Translation](https://arxiv.org/abs/2306.07650)".
![method](/egs/figs/method.png)

## Environment Configuration
```bash
conda create -n tab python=3.6
conda activate tab
conda install pytorch==1.8.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# cudatoolkit=10.2 is also acceptable.

mkdir ~/TAB && cd ~/TAB && mkdir data checkpoints
git clone https://github.com/hannlp/tab Fairseq-S2T
cd Fairseq-S2T
pip install --upgrade pip
pip install --editable ./
pip install sacrebleu==1.4.13 pandas sacremoses sentencepiece configargparse gpustat tensorboard editdistance
```

## Examples on MuST-C en-de
```bash
cd ~/TAB/Fairseq-S2T/egs/mustc/st
```

### Data Preparation
```bash
ln -s path/to/mustc_root/ende ~/TAB/data/mustc.en-de

bash run.sh --tgt_lang de
# It may take a bit of time.
```
Once you have pre-processed the data, the directory should look like this:

```bash
ls ~/TAB/data/mustc.en-de/st
config_share.yaml  dev.tsv      spm_unigram10000_st_share.model  spm_unigram10000_st_share.vocab  tst-COMMON.en
dev.en             fbank80.zip  spm_unigram10000_st_share.txt    train.tsv                        tst-COMMON.tsv
```

### Tuning with auxiliary branch
1. Download [the pre-trained ASR and MT models](https://drive.google.com/drive/folders/1e1w9UpQQ2DkQ1mMKIdgQrvLNzqQ2Yq2G?usp=sharing) and modify the parameters in `conf/tab.yaml` such as `load-pretrained-xxx-from` to match the corresponding models. 
2. Some other parameters in the paper can be adjusted, including: `use-auxiliary-branch,consistency-type,consistency-weight,replacement-probability-strategy,replacement-probability,uncertainty-gamma`.
3. Run the following command to start training (Note that `gpu_num * max_tokens * update_freq` should be 320k. And `exp_tag` can be set to mark and distinguish different experiments).
```bash
bash run.sh --stage 1 --stop_stage 1 --tgt_lang de --train_config tab --exp_tag uncertainty_gamma0.5_alpha5 --gpu_num 4 --max_tokens 20000 --update_freq 4
```
The trained models will be saved at `~/TAB/checkpoints/mustc.en-de/st/{DATE}_tab_{EXP_TAG}`

### Evaluate
```bash
bash run.sh --stage 2 --stop_stage 2 --tgt_lang de --exp_name {DATE}_tab_{EXP_TAG} --gpu_num 1 --test_subset dev,tst-COMMON
```

## Contact and Citation
To keep things simple, we have not included the scripts for the pre-training stage. If you have any questions, please do not hesitate to contact me at `hanyuchen114@gmail.com`. If you find this repository helpful, please cite it as:
```bash
@article{DBLP:journals/corr/abs-2306-07650,
  author       = {Yuchen Han and
                  Chen Xu and
                  Tong Xiao and
                  Jingbo Zhu},
  title        = {Modality Adaption or Regularization? {A} Case Study on End-to-End
                  Speech Translation},
  journal      = {CoRR},
  volume       = {abs/2306.07650},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2306.07650},
  doi          = {10.48550/arXiv.2306.07650},
  eprinttype    = {arXiv},
  eprint       = {2306.07650},
  timestamp    = {Sat, 17 Jun 2023 18:52:05 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2306-07650.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
