#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 8 #4 6 10 12
    do
        for weight in 0.9 #0.5 0.7
        do
            for weight1 in 0.5 0.6 0.7 0.8 0.9 0.95
            do
                python run.py \
                --dataset $dataset \
                --logger_name 'shark' \
                --method 'shark' \
                --data_mode 'multi-class' \
                --train \
                --save_results \
                --seed $seed \
                --gpu_id '0' \
                --video_feats_path 'video_feats.pkl' \
                --audio_feats_path 'audio_feats.pkl' \
                --text_backbone 'bert-base-uncased' \
                --config_file_name 'shark' \
                --results_file_name 'shark.csv' \
                --weight_fuse_relation $weight \
                --weight_fuse_visual_comet $weight1
            done
        done
    done
done