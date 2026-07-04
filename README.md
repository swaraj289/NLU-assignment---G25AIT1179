# Sanskrit-to-English Neural Machine Translation

Assignment 2, Natural Language Understanding — G25AIT1179 : Swaraj Mahindrakar

## Overview
A Transformer-based sequence-to-sequence model trained from scratch for
Sanskrit-to-English translation on the provided parallel corpus (10k
sentence pairs).

## Model
- 4-layer encoder / 4-layer decoder Transformer (PyTorch `nn.Transformer`)
- d_model 256, 8 attention heads, FFN dim 1024, dropout 0.1
- Joint SentencePiece BPE vocabulary (8000 merges) over both languages
- Weight tying between the embedding and output projection
- 9,429,824 parameters

## Pre-trained models / external methods disclosure
- No pre-trained weights are used in the translation model; it is trained
  from scratch on the provided training split only.
- SentencePiece is used to learn the BPE tokenizer from the training data.
- roberta-large is downloaded internally by the official `bert-score`
  library solely for computing the BERTScore evaluation metric. It plays
  no role in translation.

## Files
- `G25AIT1179_training.ipynb` — full training pipeline (data, tokenizer,
  model, training, dev/test evaluation, submission generation)
- `G25AIT1179_evaluation.ipynb` — evaluation script: loads the trained
  checkpoint, translates a given test CSV, writes `submission.csv`, and
  reports BLEU, BERTScore F1, inference time and parameter count
- `best_model.pt` — trained checkpoint (best dev loss, epoch 18/30)
- `spm_joint.model`, `spm_joint.vocab` — trained tokenizer
- `submission.csv` — predictions on the provided test set

## Running evaluation on a new test set
1. Open `G25AIT1179_evaluation.ipynb` in Colab (GPU runtime).
2. Upload `best_model.pt`, `spm_joint.model` and the test CSV files.
3. Set the `TEST_SA` / `TEST_EN` paths in the second cell.
4. Run all cells. If BERTScore raises a transformers version error,
   restart the runtime once after the install cell and re-run.

## Results (provided test set)
| Metric | Value |
|---|---|
| BLEU (NLTK, default weights) | 0.0314 |
| BERTScore F1 (rescaled) | 0.1036 |
| Inference time (1000 sentences, T4) | ~14 s |
| Parameters | 9,429,824 |
