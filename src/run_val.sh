#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=12:00:00
#$-o ../log/
#$-j y
#$-cwd
#$-m a
#$-m b
#$-m e

source ~/.bashrc
source ~/.bash_profile
conda activate presumm-scibert
source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130.1 
module load cudnn/7.5/7.5.1
python train.py \
    -task abs \
    -mode validate \
    -test_all \
    -batch_size 3000 \
    -test_batch_size 500 \
    -bert_data_path ../bert_data_survey/bert.pt_data_survey \
    -log_file ../logs/val_abs_survey \
    -model_path /scratch/acc12378ha/PreSumm-scibert/models/abs \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 50 \
    -result_path ../logs/abs_survey

