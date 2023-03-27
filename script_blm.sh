#! /usr/bin/bash

arch=roberta_tnn_v2_decay_99
# change to your data dir
data_dir=path_to_bin_data

bash train_blm.sh 8 $arch $data_dir