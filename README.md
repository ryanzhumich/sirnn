# SIRNN

This repo contains Theano implementations of the Speaker Interaction RNNs in the following paper:

[Addressee and Response Selection in Multi-Party Conversations with Speaker Interaction RNNs](https://arxiv.org/abs/1709.04005).

The code and data is based on [Addressee and Response Selection for Multi-Party Conversation](https://github.com/hiroki13/response-ranking)

## Dependencies
  - Python 2.7
  - Theano 0.9.0

## Usage
  - Static Model: `python -m adr_res_selection.main.main -mode train --train_data ../data/input/train-data.cand-2.gz --dev_data ../data/input/dev-data.cand-2.gz --test_data ../data/input/test-data.cand-2.gz --model static --data_size 100`
  - Dynamic Model: `python -m adr_res_selection.main.main -mode train --train_data ../data/input/train-data.cand-2.gz --dev_data ../data/input/dev-data.cand-2.gz --test_data ../data/input/test-data.cand-2.gz --model dynamic --data_size 100`
  - SIRNN Model: `python -m adr_res_selection.main.main -mode train --train_data ../data/input/train-data.cand-2.gz --dev_data ../data/input/dev-data.cand-2.gz --test_data ../data/input/test-data.cand-2.gz --model dynamic --data_size 100`
