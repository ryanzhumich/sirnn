# SIRNN

Theano implementations of the Speaker Interaction RNNs(SIRNN) in the following paper:

[Addressee and Response Selection in Multi-Party Conversations with Speaker Interaction RNNs](https://arxiv.org/abs/1709.04005).

The code and data is based on [Addressee and Response Selection for Multi-Party Conversation](https://github.com/hiroki13/response-ranking)

## Dependencies
  - Python 2.7
  - Theano 0.9.0

## Data
  - To run experiments with 10 candidates, concatentate train-data.cand-10.1.gz and train-data.cand-10.2.gz into train-data.cand-10.gz: `zcat train-data.cand-10.1.gz train-data.cand-10.2.gz | gzip > train-data-cand-10.gz`
  - Download [300-d Glove embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip) and save it as data/glove.840B.300d.txt

## Usage
  - SIRNN with 2 candidates and 15 context length: `cd code; ./run_interleaveGRUcollab_match_CAND2_Nc15.sh`
