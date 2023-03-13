# TNN

Official implementation of Transnormer in our ICLR 2023 paper - [Toeplitz Neural Network for Sequence Modeling](https://openreview.net/forum?id=IxmWsm4xrua).



[TOC]



## Network Architecture

The overall network architecture is as follows:

![](./network.png)



## Experiments

### Environments Preparation

Our experiment uses two conda environments, where Autoregressive language modeling, Bidirectional language modeling and Image modeling needs to configure the environment according to the Env1 part, and Long Range Arena Benchmark needs to configure the environment according to the Env2 part.

#### Env1

First build the conda environment based on the yaml file:

```
conda env create --file env1.yaml
```

If you meet error when install torch, just remove torch and torchvision in the yaml file, rerun the above command, and then run the below commands:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Finally install our version of fairseq:

```
git clone https://github.com/OpenNLPLab/fairseq-evo.git
cd fairseq
pip install --editable ./
```



#### Env2

Build the conda environment based on the yaml file:

```
conda env create --file env2.yaml
```



### Autoregressive language model

#### 1) Preprocess the data

First download the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):

```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

Next encode it with the GPT-2 BPE:

```
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```

Finally preprocess/binarize the data using the GPT-2 fairseq dictionary:

```
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```

This step comes from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).



#### 2) Train the autoregressive language model

Use the following command to train autoregressive language model:

```
bash script_alm.sh
```

You should change data_dir to the preprocessed data.



#### 3) Length extrapolation

After training, you can do length extrapolation test by the following command, where length is the test length, e.g. 512, 1024,....:

```
bash length_extrapolation.sh tnn_v2_decay_99_pre length
```





### Bidirectional language model

#### 1) Preprocess the data

The same as Autoregressive language model part.



#### 2) Train the bidirectional language model

Use the following command to train bidirectional language model:

```
bash script_blm.sh
```

You should change data_dir to the preprocessed data.



#### 3) Finetuning

Please refer to the [official Fairseq script](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md).



### Image modeling

#### 1) Preparation

Download the codebase:

```
git clone https://github.com/OpenNLPLab/im.git
```



#### 2) Training

Use the following command for training:

```
bash script_im.sh
```



### LRA

#### 1) Preparation

Download the codebase:

```
git clone https://github.com/OpenNLPLab/lra.gits
```



#### 2) Training

Use the follow script to run the exeriments, you should change `PREFIX` to your lra path, change `tasks` to a specific task, for aan, imdb and listops, the `archs` should be `tno`, for other tasks, the `archs` should be `tno2d`:

```
python script_lra.py
```



## Standalone code

For those of you who want to use tnn in your projects, you can install tnn-pytorch:

```
$ pip install tnn-pytorch
```

The code base is at the following address, you can adapt it as needed:

- [https://github.com/Doraemonzzz/tnn-pytorch](https://github.com/Doraemonzzz/tnn-pytorch)



## Citation

```
@inproceedings{
qin2023toeplitz,
title={Toeplitz Neural Network for Sequence Modeling},
author={Zhen Qin and Xiaodong Han and Weixuan Sun and Bowen He and Dong Li and Dongxu Li and Yuchao Dai and Lingpeng Kong and Yiran Zhong},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=IxmWsm4xrua}
}
```



## Wip

- [ ] Check training script.
- [ ] Update tnn-pytorch.



