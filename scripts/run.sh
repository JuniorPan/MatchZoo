#!/bin/sh
cd ../matchzoo
env CUDA_VISIBLE_DEVICES=$2 python main.py --phase train --model_file models/wikiqa_config/$1_wikiqa.config
