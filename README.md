# Word Embeddings from Scratch — Amharic Skip-gram

> `corpus_probe_7Q4M`  
> Standard library only · No NumPy · No PyTorch · No external dependencies

---

## Overview

This project implements a **Skip-gram word embedding model** entirely from scratch in pure Python, trained on a small **Amharic (Ethiopian language)** corpus. Every component — matrix operations, softmax, backpropagation, SGD, cosine similarity — is written using only Python's built-in `math`, `random`, and `json` modules.

The goal is to demonstrate the full mathematical pipeline of word embeddings without hiding any detail behind a library.

---

## File Structure

```
word-embedding/
├── skipgram.py              # Complete implementation
├── amharic_skipgram.json    # Saved model (generated after first run)
└── README.md                # This file
```

---

## How to Run

```bash
python3 skipgram.py
```

No installation required. Python 3.7+ is sufficient.

---

## Architecture

The model uses two matrices:

| Matrix | Shape  | Role |
|--------|--------|------|
| `E`    | V × d  | Input embedding — one row per vocabulary word |
| `U`    | d × V  | Output matrix — maps hidden vector to vocabulary scores |

For each `(center, context)` training pair:

1. Look up the center word's row in `E` → hidden vector `h`
2. Multiply `h` by `U` → score vector `s` of length V
3. Apply softmax to `s` → probability distribution `p`
4. Compute cross-entropy loss `L`
5. Backpropagate gradients to `U` and to row `E[i]` only
6. Update with SGD

---

## Mathematical Formulation

### Notation

| Symbol | Meaning |
|--------|---------|
| V | Vocabulary size |
| d | Embedding dimension |
| i | Center word index |
| t | Target context word index |

### Forward Pass

```
h      = E[i]                          # embedding lookup
s[j]   = sum_k  h[k] * U[k][j]        # score for word j
```

### Numerically Stable Softmax

```
m      = max(s)
a[j]   = exp(s[j] - m)
p[j]   = a[j] / sum(a)
```

Subtracting `m` before `exp` prevents overflow for large scores. The probabilities are mathematically identical because the constant cancels.

### Loss (Cross-Entropy)

```
L = (m - s[t]) + ln(sum(a))
```

This is equivalent to `-log(p[t])` but reuses the stable softmax intermediates.

### Mean Dataset Loss

```
J = (1/N) * sum_{n=1}^{N} L_n
```

Evaluated with frozen weights after each epoch.

### Error Signal

```
e[j] = p[j] - 1   if j == t
       p[j]        otherwise
```

### Gradients

```
grad_U[k][j] = h[k] * e[j]
grad_h[k]    = sum_j  U[k][j] * e[j]
```

Both gradients are computed using the **old** weights before any update.

### SGD Update

```
U[k][j]  ←  U[k][j]  - lr * grad_U[k][j]
E[i][k]  ←  E[i][k]  - lr * grad_h[k]
```

Only row `E[i]` (the center word) is updated. All other rows of `E` remain unchanged for that pair.

---

## Corpus

60 short Amharic sentences across six everyday topic groups:

| Group | Topic | Example sentence | Translation |
|-------|-------|-----------------|-------------|
| 1 | Movement / school | ልጁ ወደ ትምህርት ቤት ሄደ | The boy went to school |
| 2 | Eating | አባቴ ዳቦ በላ | My father ate bread |
| 3 | Drinking | ድመቷ ወተት ጠጣች | The cat drank milk |
| 4 | Cooking / baking | እናቴ ዳቦ ጋገረች | My mother baked bread |
| 5 | Reading / studying | ልጁ መጽሐፍ አነበበ | The boy read a book |
| 6 | Playing / resting | ልጅቷ ተጫወተች | The girl played |

Subjects used: ልጁ (boy), ልጅቷ (girl), አባቴ (father), እናቴ (mother), ውሻው (dog), ድመቷ (cat).

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 10 |
| Context window | 2 |
| Learning rate | 0.05 |
| Epochs | 100 |
| Random seed | 42 |

---

## Implementation Checks

Six unit tests run automatically before training, each with tolerance `1e-5`:

### Check 1 — Context Window
Sentence `[a, b, c, d]` with `window=1` must produce exactly:
```
(a,b), (b,a), (b,c), (c,b), (c,d), (d,c)
```
Also verifies no pairs cross sentence boundaries.

### Check 2 — Forward Pass
```
Vocabulary : [cats, eat, food]
E = [[1,0],[0,1],[1,1]]
U = [[0,1,-1],[1,0,1]]
Pair = (cats, eat)

Expected scores: [0, 1, -1]
```

### Check 3 — Probabilities and Loss
```
Expected p ≈ [0.244728, 0.665241, 0.090031]
Expected L ≈ 0.407606
Probabilities are non-negative and sum to 1.
```

### Check 4 — Gradients
```
Expected grad_U ≈ [[0.244728, -0.334759, 0.090031],
                   [0,        0,          0       ]]

Expected grad_h ≈ [-0.424790, 0.334759]
```

### Check 5 — One SGD Update (lr = 0.1)
```
Expected E[cats] ≈ [1.042479, -0.033476]
Other E rows must remain unchanged.
Recomputed loss ≈ 0.361859
```

### Check 6 — Numerical Stability
```
Scores [1000, 0], target index 1 → stable loss ≈ 1000
Adding 100 to all scores → probabilities and loss unchanged
```

---

## Training Results

```
Vocabulary size   : 36
Training pairs    : 476
Loss before training : 3.583510
Epoch   1 loss       : 3.583447
Epoch  10 loss       : 3.160212
Epoch  20 loss       : 2.626106
Epoch  50 loss       : 2.256451
Epoch 100 loss       : ~2.20
```

The larger vocabulary (36 words) and richer pair set (476 pairs) give the model more co-occurrence signal compared to the original 20-sentence corpus.

---

## Nearest-Neighbor Queries

Cosine similarity is computed manually:

```
cosine(a, b) = sum(a[k]*b[k]) / (sqrt(sum(a[k]^2)) * sqrt(sum(b[k]^2)))
```

Results after 100 epochs (top 3 neighbors from matrix `E`):

| Query | Neighbor 1 | Neighbor 2 | Neighbor 3 |
|-------|-----------|-----------|-----------|
| ልጁ (boy) | ውሻው 0.647 | ሄዶ 0.558 | ሄደች 0.551 |
| ምግብ (food) | ፍሬ 0.781 | ዳቦ 0.543 | በልቶ 0.370 |
| ቡና (coffee) | ወተት 0.863 | ጠጥታ 0.421 | ጋዜጣ 0.399 |
| ወደ (to/towards) | ትምህርት 0.520 | ተጫወተች 0.457 | ውስጥ 0.412 |
| ሄደ (he went) | ሄደች 0.990 | ሄዳ 0.717 | ሄዶ 0.630 |

Notable patterns:
- `ሄደ` (he went) and `ሄደች` (she went) have cosine **0.990** — even stronger clustering of masculine/feminine verb forms with the larger corpus
- `ቡና` (coffee) and `ወተት` (milk) score **0.863** — both drinks appear in nearly identical sentence frames across 60 sentences
- `ምግብ` (food) and `ፍሬ` (fruit) cluster together as consumable nouns in eating contexts
- `ሄደ / ሄደች / ሄዳ / ሄዶ` all cluster tightly — the model groups all motion-verb inflections correctly

---

## Save / Load

The model is serialised to JSON after training:

```python
save_model("amharic_skipgram.json", w2i, i2w, E, U)
w2i, i2w, E, U = load_model("amharic_skipgram.json")
```

Nearest-neighbor rankings are verified to be identical before and after reload.

---

## Numerical Gradient Check

Finite-difference verification with `epsilon = 1e-5`:

```
[L(theta + epsilon) - L(theta - epsilon)] / (2 * epsilon)
```

Results:
```
grad_U[0][0]: analytic=0.24472847  numeric=0.24472847  diff=1.47e-13
grad_h[0]  : analytic=-0.42478962  numeric=-0.42478962  diff=5.33e-12
```

Differences are at floating-point machine precision (~1e-12), confirming the analytic gradients are correct.

---

## Limitations

- **Corpus size**: 60 sentences is still small for reliable semantic generalisation. Neighbors reflect surface co-occurrence patterns, not true meaning.
- **Morphology**: Amharic is a highly inflected language. A production model would need morphological segmentation (e.g., splitting verb stems from suffixes) before tokenisation.
- **No subsampling**: Frequent words like `ወደ` (to) are not downsampled, so they dominate training signal.
- **SGD without decay**: The learning rate is fixed, causing loss oscillation in later epochs. A schedule or Adam would stabilise convergence.
- **Lower loss ≠ better semantics**: The model can overfit co-occurrence statistics of this tiny corpus without learning transferable word meaning.

---

## Function Reference

| Function | Description |
|----------|-------------|
| `preprocess(corpus)` | Splits sentence strings into token lists |
| `build_vocab(sentences)` | Builds sorted `w2i` / `i2w` dictionaries |
| `make_pairs(sentences, w2i, window)` | Generates `(center_id, context_id)` pairs within sentence boundaries |
| `init_matrix(rows, cols, seed)` | Initialises a matrix with small uniform random values |
| `forward(E, U, i)` | Computes `h = E[i]` and `s = hU` |
| `softmax(s)` | Numerically stable softmax; returns `p`, `a`, `m`, `total` |
| `loss_one(s, t)` | Cross-entropy loss for one pair |
| `gradients(h, U, p, t)` | Computes `grad_U` and `grad_h` using old weights |
| `update(E, U, i, grad_U, grad_h, lr)` | SGD step; only updates `E[i]` and all of `U` |
| `mean_loss(E, U, pairs)` | Evaluates `J = (1/N) Σ L_n` with frozen weights |
| `train(E, U, pairs, lr, epochs)` | Full training loop with per-epoch shuffling and loss reporting |
| `cosine(a, b)` | Manual cosine similarity; safe for zero-norm vectors |
| `nearest_neighbors(E, w2i, i2w, query, n)` | Prints top-n neighbors by cosine similarity |
| `save_model(path, ...)` | Serialises vocabulary and matrices to JSON |
| `load_model(path)` | Deserialises model from JSON |
| `numerical_grad_check(E, U, i, t)` | Finite-difference gradient verification |

---

## Requirements

- Python 3.7+
- No third-party packages
