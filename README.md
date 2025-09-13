# HNSW (Hierarchical Navigable Small World) — Minimal Educational Implementation

> This repository accompanies my article **“Building HNSW from Scratch”**.
> The goal is to provide a small, readable reference implementation you can run, tweak, and learn from—not a production-ready ANN engine.

---

## What is this?

This is a from-scratch, teaching-oriented implementation of a tiny HNSW-style index for approximate nearest neighbor (ANN) search. It shows:

* Probabilistic **layer assignment** (coin-flip tower)
* **Cosine similarity** scoring
* A simple **multi-layer beam search**
* **Bidirectional connections** with a max degree and basic eviction of weak links
* A small **demo** that builds an index and runs a query

The code is contained in a single file for clarity.

---

## Quick start

### 1) Requirements

* Python 3.9+
* `numpy`

```bash
pip install numpy
```

### 2) Run the demo

```bash
python main.py
```

What it does:

* Builds an index
* Inserts a few sample documents with random 64-D embeddings
* Runs a query and prints the **top-k** results

You’ll see output like:

```
Building HNSW Vector Database...
Inserting 8 documents...
  Document 0: layers [0, 1]
  Document 1: layers [0]
  ...
Index statistics:
  Total vectors: 8
  Highest layer: 2

Testing search...
Top 3 search results:
  1. (similarity: 0.783) Machine learning is transforming technology...
  2. (similarity: 0.701) Vector databases enable semantic search...
  3. (similarity: 0.655) Python is a versatile programming language...
```

*(Your similarities will vary—embeddings are random in the demo.)*

---

## How it works (in brief)

* **Layers via coin flips**: `_assign_layer()` repeatedly flips a biased coin (p=0.5) to decide how “high” a node goes. Few nodes reach the top; most remain at lower layers.
* **Search**: `search()` starts at the highest layer, keeps a small beam of good candidates, and descends layer by layer—expanding neighbors and re-ranking by cosine similarity until layer 0.
* **Connections**: `_make_connections()` links the new node to its best candidates on each layer. If a neighbor is full, it **evicts** its weakest existing link if the new link is better.
* **User API**: `insert(embedding, document)` adds data; `query(query_embedding, k)` returns the top-k matches (document, similarity, id).

---

## Using the class in your own code

```python
from hnsw import HNSW
import numpy as np

index = HNSW(max_connections=16)

# Insert your own vectors
doc = "Example document"
vec = np.random.rand(64)  # Replace with your real embedding
vid = index.insert(vec, doc)

# Query
q = np.random.rand(64)    # Replace with your real query embedding
results = index.query(q, k=5)

for r in results:
    print(r["similarity"], r["document"])
```

> In real applications, replace the random vectors with embeddings from your favorite model (e.g., OpenAI, Jina, Cohere, etc.).

---

## API overview

* `HNSW(max_connections=16)`
  Create an index with a max number of neighbors per node (per layer).

* `insert(embedding: np.ndarray, document: Any) -> int`
  Adds a vector + payload (document) and returns its `vector_id`.

* `query(query_embedding: np.ndarray, k: int = 5) -> List[dict]`
  Returns top-k dicts with `document`, `similarity`, and `vector_id`.

* `search(query_embedding: np.ndarray, num_candidates: int = 50) -> Dict[int, List[int]]`
  Educational helper that exposes per-layer candidate exploration.

---

## Notes & limitations

* **Education first**: This code aims for clarity over speed.
* **No persistence**: Everything is in memory; no save/load.
* **Random demo embeddings**: Swap in real embeddings for meaningful results.
* **Simplified HNSW**: Real HNSW has additional heuristics (e.g., M/Mmax per layer, efConstruction/efSearch tuning, heuristic neighbor selection). Here, logic is intentionally minimal so you can understand each step.

---
