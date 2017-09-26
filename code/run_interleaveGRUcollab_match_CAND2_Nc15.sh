#! /bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 \
python -m adr_res_selection.main.main \
       -mode train \
       --train_data ../data/input/train-data.cand-2.gz \
       --dev_data ../data/input/dev-data.cand-2.gz \
       --test_data ../data/input/test-data.cand-2.gz \
       --model interleavecollab \
       --unit gru \
       --batch 128 \
       --dim_emb 300 \
       --init_emb data/glove.840B.300d.txt \
       --n_prev_sents 15 \
       --crosstest 4 \
       --save 1 \
       --output_fn model_collab_CAND2_Nc15.pkl.gz \
       | tee interleaveGRUcollab_match_CAND2_Nc15.txt
