arch=tnn_v2_decay_99_pre
name=name
# change to your data dir
data_dir=$1

bash train.sh 8 $arch $name 0.0005 0.0 M3T 0.2 $data_dir