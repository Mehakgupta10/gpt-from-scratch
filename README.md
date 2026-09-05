# Building a GPT from Scratch

A character-level GPT (Generative Pre-trained Transformer) built from raw Python and PyTorch — no pre-built model libraries, no shortcuts. This project follows the architecture used in GPT-2/GPT-3, implemented step by step to actually understand how self-attention and transformers work under the hood.

## What it does

Trained on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1.1M characters), the model learns to predict the next character in a sequence, one at a time. After training, it generates text that mimics Shakespeare's style — character names, dialogue formatting, and sentence rhythm — despite never being taught a single rule of English grammar.

## Architecture

Built entirely from scratch, piece by piece:

- **Tokenization** — character-level vocabulary (65 unique characters)
- **Self-attention** — implemented from the ground up, including the causal masking trick (via matrix multiplication with a triangular matrix) that lets each token attend only to previous tokens
- **Multi-head attention** — multiple attention heads running in parallel
- **Feed-forward layers** — per-token computation after attention
- **Residual connections + Layer Normalization** — for stable training in a deep network
- **Full Transformer blocks** stacked into a complete GPT language model

## Results

Baseline comparison:

| Model | Context length | Final loss |
|---|---|---|
| Bigram (1-character memory) | 1 | 2.31 |
| GPT (self-attention) | 32 | 2.05 |

Sample output after training: 
Not real English yet — this is a small model trained quickly on CPU — but the structure (character names, colons, dialogue-like formatting) is clearly emerging, a big step up from the bigram baseline.

## What I learned

- How tokenization converts text into a form neural networks can process
- The core mathematical trick behind self-attention (weighted aggregation via a lower-triangular matrix)
- How queries, keys, and values let tokens "communicate" with each other
- Why residual connections and layer normalization are necessary for training deep networks
- How all of these pieces combine into the same architecture used by GPT-2, GPT-3, and GPT-4 — the difference between this and those models is almost entirely scale, not architecture

## Running it

```bash
pip install torch jupyter
jupyter notebook
```

Open `gpt-from-scratch.ipynb` and run the cells in order.
