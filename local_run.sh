#!/bin/bash

dt=$(date "+%Y%m%d%H%M")
pred_dir=preds_${dt}_$1
score_dir=scores_${dt}_$1
mkdir $pred_dir
mkdir $score_dir

python -u ingestion/ingestion.py --dataset_dir=./dev_data --code_dir=./$1 --output_dir=./${pred_dir} |& tee ./${pred_dir}/results_ingestion.log

python -u scoring/score.py --dataset_dir=./dev_data --prediction_dir=./${pred_dir} --score_dir=./${score_dir} |& tee ./${score_dir}/results_scoring.log
