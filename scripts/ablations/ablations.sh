#!/usr/bin/env bash

python ablations.py --models gpt_j --datasets all --settings all --abl_types all --num_inputs 1000 --num_demos max & python ablations.py --models gpt2_xl,gpt_neox --datasets all --settings all --abl_types attention,mlp --num_inputs 1000 --num_demos max