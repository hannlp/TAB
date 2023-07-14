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
@inproceedings{han-etal-2023-modality,
    title = "Modality Adaption or Regularization? A Case Study on End-to-End Speech Translation",
    author = "Han, Yuchen  and
      Xu, Chen  and
      Xiao, Tong  and
      Zhu, Jingbo",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.115",
    pages = "1340--1348",
    abstract = "Pre-training and fine-tuning is a paradigm for alleviating the data scarcity problem in end-to-end speech translation (E2E ST). The commonplace {''}modality gap{''} between speech and text data often leads to inconsistent inputs between pre-training and fine-tuning. However, we observe that this gap occurs in the early stages of fine-tuning, but does not have a major impact on the final performance. On the other hand, we find that there has another gap, which we call the {''}capacity gap{''}: high resource tasks (such as ASR and MT) always require a large model to fit, when the model is reused for a low resource task (E2E ST), it will get a sub-optimal performance due to the over-fitting. In a case study, we find that the regularization plays a more important role than the well-designed modality adaption method, which achieves 29.0 for en-de and 40.3 for en-fr on the MuST-C dataset.",
}
