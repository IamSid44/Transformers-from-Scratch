# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants, and BLT

Roll number **2023102040**. Encoder–decoder Transformer built from elementary PyTorch
operations (no `nn.Transformer`, no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`),
used to run a controlled five-way ablation on decrypting XOR-enciphered binary sequences into
English plaintext.

## Links

**Weights & Biases** — project: <https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040>

| Config | Run |
|---|---|
| C1 | https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040/runs/ykhjp07w |
| C2 | https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040/runs/x4t4740q |
| C3 | https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040/runs/fhlb49xz |
| C4 | https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040/runs/wabl8zj2 |
| C5 | https://wandb.ai/iamsid44-iiit-hyderabad/anlp-a1-2023102040/runs/halcr0ju |

**HuggingFace checkpoints** — one repo per configuration, each holding `best.pt`, both
tokenizers and a model card:

| Config | Repo |
|---|---|
| C1 | https://huggingface.co/siddarthg44/anlp-a1-2023102040-C1 |
| C2 | https://huggingface.co/siddarthg44/anlp-a1-2023102040-C2 |
| C3 | https://huggingface.co/siddarthg44/anlp-a1-2023102040-C3 |
| C4 | https://huggingface.co/siddarthg44/anlp-a1-2023102040-C4 |
| C5 | https://huggingface.co/siddarthg44/anlp-a1-2023102040-C5 |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `brown_cipher.txt` and `brown_plain.txt` in the project root (the two corpus files are
not shipped here; they are 27 MB together).

Optional, for the W&B and HuggingFace integrations, create a `.env` in the project root:

```
WANDB_API_KEY=...
HF_TOKEN=...
HF_USERNAME=...
```

Without `WANDB_API_KEY` training logs offline to `./wandb/`; without the HF variables `--push`
is skipped. Neither is needed to train or evaluate.

## Running

```bash
# 1. corpus checks + train both BPE tokenizers (writes outputs/tokenizers/)
python src/dataset.py

# 2. train the entropy model C5's dynamic patching depends on (writes outputs/entropy_lm.pt)
python src/models/entropy_lm.py

# 3. train one configuration, or all five
python src/train.py --config C1 --push
python src/train.py --all --push

# 4. evaluate (greedy decoding on the test split)
python src/train.py --evaluate --all

#    ... or evaluate several in parallel without racing the shared results files
python src/train.py --evaluate --config C1 --no-write   # per-config metrics_C1.json
python src/train.py --collect                           # merge + redraw figures

# 5. cost table, one configuration per process on an idle GPU
python src/profile_cost.py --all

# 6. regenerate the report tables and figures
python src/report_assets.py
```

Self-tests, each runnable on its own:

```bash
python src/dataset.py            # split integrity, chunk reassembly, BPE round-trip
python src/models/blt.py         # BLT shapes, causality, patch grid, one-batch overfit
python src/verify_tokenizer.py   # differential test of src/bpe.py against HuggingFace tokenizers
```

## Layout

```
2023102040_assignment1/
├── src/
│   ├── models/
│   │   ├── attention.py      # scaled dot-product attention, MHA, GQA, encoder/decoder layers
│   │   ├── positional.py     # sinusoidal absolute and RoPE
│   │   ├── norm.py           # LayerNorm and RMSNorm
│   │   ├── blt.py            # local encoder/decoder, patch pooler, BLTSeq2Seq
│   │   └── entropy_lm.py     # byte-level entropy model + BLT's dynamic patching rule
│   ├── dataset.py            # corpus, splits, chunking, tokenized and token-free loaders
│   ├── train.py              # configurations, training loop with WandB, evaluation
│   ├── utils.py              # metrics and plots
│   ├── bpe.py                # BPE implemented from scratch (no `tokenizers` library)
│   ├── profile_cost.py       # isolated per-configuration cost measurement
│   ├── report_assets.py      # tables and figures for the report
│   ├── recover_history.py    # rebuild a history JSON from a training log
│   └── verify_tokenizer.py   # differential test of bpe.py against the library
├── outputs/
│   ├── figures/              # loss_curves, metric_comparison, memory_speed, patching
│   ├── logs/                 # per-configuration training and evaluation logs
│   ├── tokenizers/           # the two trained BPE vocabularies
│   ├── entropy_lm.pt         # the entropy model + its calibrated threshold
│   ├── history_C*.json       # per-epoch loss, timing, memory, source-dependence
│   ├── samples_C*.txt        # greedy-decoding samples
│   ├── results.csv/.json     # test metrics, per chunk and per reassembled line
│   ├── profile.json          # cost, measured one configuration per process
│   ├── runtime_stats.json    # training-time throughput (contended; see note below)
│   └── REPORT_MATERIALS.md   # every number in the report, with its provenance
├── README.md
└── Report.pdf
```

Two files sit outside the tree the brief specifies, because the code does not run without
them: **`src/bpe.py`** (BPE from scratch — the assignment forbids prebuilt tokenizers, so this
replaces the `tokenizers` library and exposes the same API) and **`src/models/entropy_lm.py`**
(the byte-level LM that drives C5's entropy-based dynamic patching). `profile_cost.py`,
`report_assets.py`, `recover_history.py` and `verify_tokenizer.py` are tooling, not part of the
model.

## The five configurations

| Cfg | Changed from base | Positional | Attention | Norm | Tokenization |
|---|---|---|---|---|---|
| C1 | — (base) | Sinusoidal absolute | MHA (8 heads) | LayerNorm | BPE subword |
| C2 | Positional encoding | **RoPE** | MHA (8 heads) | LayerNorm | BPE subword |
| C3 | Attention | Sinusoidal absolute | **GQA** (8q / 2kv) | LayerNorm | BPE subword |
| C4 | Normalization | Sinusoidal absolute | MHA (8 heads) | **RMSNorm** | BPE subword |
| C5 | Tokenization | Sinusoidal absolute | MHA (8 heads) | LayerNorm | **BLT (token-free)** |

C2–C5 are built from C1 by `dataclasses.replace`, so it is structurally impossible for more
than the named field to differ. All five share `d_model` 256, 4+4 layers, `d_ff` 2048, dropout
0.1, Adam at 6e-4 with 1 epoch warmup then cosine decay to 10% of peak, gradient clipping 1.0,
label smoothing 0.1, batch size 256, seed 42.

## Results

Test set, greedy decoding, per decoded 32-character chunk:

| Cfg | Bit % | Char % | Seq % | Lev ↓ | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 99.58 | 98.95 | 95.34 | 0.067 | 98.10 | 99.08 |
| C2 | **99.73** | **99.34** | **96.86** | **0.047** | **98.74** | **99.36** |
| C3 | 99.50 | 98.77 | 95.26 | 0.075 | 98.04 | 99.05 |
| C4 | 99.60 | 99.03 | 96.25 | 0.059 | 98.47 | 99.27 |
| C5 | 98.81 | 97.32 | 80.38 | 0.727 | 90.71 | 95.07 |

Cost, measured one configuration per process on an otherwise idle GPU:

| Cfg | Params (M) | s/step | s/epoch | Train MiB | ms/chunk | Infer MiB |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 12.89 | 0.764 | 232 | 2952 | 3.84 | 294 |
| C2 | 12.89 | 0.872 | 264 | 2953 | 6.05 | 294 |
| C3 | 11.70 | 0.783 | 237 | 2937 | 4.69 | 280 |
| C4 | 12.88 | 0.726 | 220 | 2781 | 3.98 | 294 |
| C5 | 26.68 | 1.459 | 442 | 9600 | 37.32 | 729 |

Use `outputs/profile.json` for anything cost-related. The `sec_per_epoch` and
`examples_per_sec` fields in `runtime_stats.json` come from the training runs, which were
executed concurrently and are therefore contended.

## Two things to know when reading these numbers

**C5 ran 20 epochs against C1–C4's 40.** Training was stopped on a compute-budget constraint,
not by early stopping: its validation loss improved on each of its last four epochs
(0.1263 → 0.1239 → 0.1230 → 0.1212) and its cosine schedule, defined over 40 epochs, was
truncated at ~60% and never annealed. C5's quality numbers are a lower bound.

**Per-token and per-byte cross-entropy are not comparable.** C1–C4 score one BPE token at a
time over a 4,096-way softmax; C5 scores one byte at a time over a 259-way one, and predicts
more units per chunk. `outputs/figures/loss_curves.png` therefore carries a third panel in
bits per character, the one unit both pathways share, which is the panel to read across all
five.
