# NiuTrans-Fairseq-S2T

This project adapts the [fairseq](https://github.com/pytorch/fairseq) toolkit for speech-to-text tasks, including speech recognition and speech translation.

It contains the implementation of the following methods proposed by NiuTrans Team.

[Stacked Acoustic-and-Textual Encoding: Integrating the Pre-trained Models into Speech Translation Encoders](https://arxiv.org/abs/2105.05752)

## Key Features

### Training

- Support the Kaldi-style complete recipes
- ASR, MT, and ST pipelines (bin)
- Read training config in yaml file
- CTC multi-task learning
- MT training in the ST-like way (Online tokenizer) (This may be slowly.)
- speed perturb during pre-processing
  
### Model

- Conformer Architecture
- Load pre-trained modules
- Relative position representation
- Stacked acoustic-and-textual encoding
- Progressive down-sampling for acoustic encoding

## Installation

* Note we only test the following environment.

1. Python == 3.6
2. torch == 1.8, torchaudio == 0.8.0, cuda == 10.2
3. apex
```
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
4. nccl
```
make -j src.build CUDA_HOME=<path to cuda install>
```
5. gcc ≥ 4.9 (We use the version 5.4)
6. python library 
```
pip install pandas sentencepiece configargparse gpustat tensorboard editdistance
```

## Code Structure

We supply the recipes for multiple benchmarks in the egs folder, including machine translation, speech recognition, and speech translation corpora.
Besides, we also provide the template for other benchmarks.

Here is an example for MuST-C:

```markdown
mustc
├── asr
│   ├── binary.sh
│   ├── conf/
│   ├── decode.sh
│   ├── local/
│   ├── run.sh
│   └── train.sh
├── mt
│   ├── binary.sh
│   ├── conf/
│   ├── decode.sh
│   ├── local/
│   ├── run.sh
│   └── train.sh
└── st
    ├── binary.sh
    ├── conf/
    ├── decode.sh
    ├── local/
    ├── run.sh
    └── train.sh
```

* run.sh: the core script that includes the whole pipeline
* train.sh: call the run.sh for training
* decode.sh: call the run.sh for decoding
* binary.sh: generate the datasets alone
* conf: the folder to save the configure files (.yaml). 
* local: the folder to save utils
  * monitor.sh: check the GPUS for running the program automatically 
  * parse_options.sh: parse the parameters for run.sh
  * utils.sh: the util shell functions
  
## Citations

```bibtex
@inproceedings{xu-etal-2021-stacked,
    title = "Stacked Acoustic-and-Textual Encoding: Integrating the Pre-trained Models into Speech Translation Encoders",
    author = "Xu, Chen  and
      Hu, Bojie  and
      Li, Yanyang  and
      Zhang, Yuhao  and
      Huang, Shen  and
      Ju, Qi  and
      Xiao, Tong  and
      Zhu, Jingbo",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.204",
    doi = "10.18653/v1/2021.acl-long.204",
    pages = "2619--2630",
}
```