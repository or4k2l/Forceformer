# Forceformer Reproduction and Follow-up Experiments

This directory contains a standalone PyTorch reproduction of the throughput-versus-accuracy
experiment in the repository's original notebook, plus exploratory cascade and sparse-routing
follow-ups. It is a separate CPU-oriented run on a small sentiment corpus; its numbers must not
be mixed with or presented as a direct replication of the original IMDB/T4 measurements.

## Reproduction differences

- **Dataset:** NLTK's `movie_reviews` corpus (2,000 reviews), rather than IMDB's 50,000 reviews.
  The code downloads the corpus on first use into `~/.cache/forceformer-repro/nltk_data`.
  Set `NLTK_DATA` to use a different cache directory.
- **Hardware and length:** the recorded run used CPU and trained at sequence length 256, rather
  than a Tesla T4 GPU and length 512.
- **Validation size:** the fixed-seed, non-stratified validation split contains only 200 reviews.
  Small accuracy differences are noisy; the scores are not evidence of a general model advantage.
- **Throughput:** scripts use synthetic token IDs and run the model's embedding and mixer on them.
  Tokenization, data loading, and model loading are excluded. Results depend on hardware, batch
  size, software versions, warm-up, and system load.

## Models

| Model | Description | Sequence-length complexity |
|---|---|---|
| `glu` | Element-wise GLU mixer; no cross-token communication | O(L) |
| `linear_attn` | Bidirectional feature-map linear attention | O(L) |
| `softmax_attn` | Standard multi-head softmax attention | O(L^2) |
| `gated_fusion` | Linear attention blended with local attention (radius 8; width 17) | O(L) for fixed window |

`gated_fusion` is an exploratory implementation assembled from familiar components, not a claim
of a validated novel architecture. Its sparse inference mode selects a fraction of token positions
for local-attention refinement; it uses weights trained with the dense path and is an approximation
at inference time, not a separately trained sparse model.

## Setup and rerunning

From this directory, create a virtual environment and install the experiment-specific dependencies:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python data.py
python train.py
python throughput.py
python throughput_sparse.py
python eval_sparse.py
python cascade.py
```

`train.py` trains all four models by default and writes one checkpoint per model plus
`train_results.json`. Select a subset, device, or training parameters with, for example:

```bash
python train.py --models glu linear_attn --device cpu --epochs 10 --seed 42
```

`throughput.py` requires the checkpoints from `train.py`; it measures all four models and writes
`throughput_results.json` and `long_context_results.json`. The sparse and cascade scripts also use
the trained checkpoints. The initial corpus download requires network access. All outputs are
written beside the scripts, regardless of the shell's current directory.

## Recorded results (historical snapshot)

The JSON files in this folder preserve the supplied run's measurements. The base comparison at
length 512 was:

| Model | Validation accuracy | Throughput (sequences/s) | Accuracy x throughput |
|---|---:|---:|---:|
| GLU | 61.5% | 783.9 | 482.1 |
| Linear attention | 66.0% | 522.7 | 345.0 |
| Softmax attention | 64.5% | 46.7 | 30.1 |

The base table reports the best validation accuracy observed over training epochs. The supplied
historical checkpoints can score differently from that best-epoch value (the saved softmax checkpoint
scores 63.5%, versus the recorded best of 64.5%). New runs save the best-validation checkpoint.

The supplied sparse-routing snapshot reports 60.0% validation accuracy and 543.0 sequences/s at
a 10% routing fraction, versus 63.0% and 195.7 sequences/s for the dense path. These are
exploratory single-run measurements on a small validation set, not statistically established
improvements. Rerunning scripts overwrites the corresponding result files and will not necessarily
reproduce identical timings or scores.

The cascade calculation estimates effective throughput from separately measured single-model
rates; it is not a direct end-to-end latency benchmark. Long-context runs are inference stress
tests only and do not measure accuracy at those lengths. Read the scripts and raw JSON alongside
the summary before interpreting the values.

## Files

- `data.py`, `models.py`, `train.py`: data preparation, architectures, and training.
- `throughput.py`: base throughput sweep and long-context stress test.
- `throughput_sparse.py`, `eval_sparse.py`: sparse inference timing and accuracy sweep.
- `cascade.py`: confidence-based routing estimate.
- `*.pt`: supplied model checkpoints; `*_results.json`: supplied result snapshots.

The repository is licensed under Apache-2.0; see the top-level `LICENSE`.

## Dataset citation

The `movie_reviews` corpus originates from Pang and Lee, "A Sentimental Education: Sentiment Analysis
Using Subjectivity Summarization Based on Minimum Cuts," ACL 2004.
