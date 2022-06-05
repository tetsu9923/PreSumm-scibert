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
python train.py -task abs \
    -mode train \
    -bert_data_path ../bert_data_survey/bert.pt_data_survey \
    -dec_dropout 0.2 \
    -model_path /scratch/acc12378ha/PreSumm-scibert/models/abs \
    -sep_optim true \
    -lr_bert 0.002 \
    -lr_dec 0.2 \
    -save_checkpoint_steps 1000 \
    -batch_size 140 \
    -train_steps 50000 \
    -report_every 50 \
    -accum_count 5 \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps_bert 20000 \
    -warmup_steps_dec 10000 \
    -max_pos 512 \
    -visible_gpus 0 \
    -log_file ../logs/abs_survey

