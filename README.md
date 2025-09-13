# HNSW Algorithm Implementation

A Python implementation of the **Hierarchical Navigable Small World (HNSW)** algorithm for efficient approximate nearest neighbor search in high-dimensional vector spaces.

## What is HNSW?

HNSW (Hierarchical Navigable Small World) is a state-of-the-art algorithm for approximate nearest neighbor search. It creates a multi-layer graph structure where:

- **Layer 0** contains all vectors
- **Higher layers** contain progressively fewer vectors  
- **Search** starts from the top layer and navigates down to find similar vectors efficiently

This approach provides excellent performance for similarity search in vector databases, making it ideal for applications like semantic search, recommendation systems, and machine learning.

## Features

✅ **Multi-layer graph structure** for efficient search  
✅ **Probabilistic layer assignment** using coin-flip method  
✅ **Bidirectional connections** between nodes  
✅ **Connection limit enforcement** with neighbor eviction  
✅ **Cosine similarity** for vector comparison  
✅ **Beam search** optimization through layers  
✅ **User-friendly query interface** returning top-k results  

## Installation

### Requirements
- Python 3.6+
- NumPy

### Install Dependencies
```bash
pip install numpy
```

## Usage

### Basic Example

```python
import numpy as np
from main import HNSW

# Create HNSW index
index = HNSW(max_connections=16)

# Insert vectors with associated documents
embedding1 = np.random.rand(64)  # 64-dimensional vector
doc_id1 = index.insert(embedding1, "First document")

embedding2 = np.random.rand(64)
doc_id2 = index.insert(embedding2, "Second document")

# Query for similar vectors
query_vector = np.random.rand(64)
results = index.query(query_vector, k=5)

# Results contain similarity scores and documents
for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Document: {result['document']}")
    print(f"Vector ID: {result['vector_id']}")
```

### Running the Demo

The repository includes a demonstration with sample documents:

```bash
python main.py
```

This will:
1. Create an HNSW index with sample documents
2. Show the multi-layer structure for each inserted vector
3. Perform a search query and display top results

## API Reference

### HNSW Class

#### Constructor
```python
HNSW(max_connections=16)
```
- `max_connections`: Maximum number of connections per node in the graph

#### Methods

##### `insert(embedding, document)`
Inserts a new vector into the index.
- `embedding`: NumPy array representing the vector
- `document`: Associated document/data
- **Returns**: Vector ID (integer)

##### `query(query_embedding, k=5)`
Searches for the k most similar vectors.
- `query_embedding`: NumPy array to search for
- `k`: Number of results to return
- **Returns**: List of dictionaries with `document`, `similarity`, and `vector_id`

##### `search(query_embedding, num_candidates=50)`
Low-level search returning candidates by layer.
- `query_embedding`: NumPy array to search for  
- `num_candidates`: Number of candidates to consider
- **Returns**: Dictionary mapping layer numbers to candidate lists

##### `cosine_similarity(vec1, vec2)`
Calculates cosine similarity between two vectors.
- **Returns**: Similarity score between -1 and 1

## How It Works

1. **Layer Assignment**: New vectors are probabilistically assigned to layers using coin flips
2. **Search for Neighbors**: For insertion, the algorithm searches existing vectors to find potential neighbors
3. **Connection Creation**: Bidirectional connections are created between similar vectors
4. **Neighbor Eviction**: When connection limits are exceeded, the least similar neighbors are removed
5. **Multi-layer Search**: Queries start from the highest layer and beam search down to layer 0

## Demo Output Example

```
Building HNSW Vector Database...
Inserting 8 documents...
  Document 0: layers [0]
  Document 1: layers [0]
  Document 2: layers [0, 1]
  Document 3: layers [0]
  Document 4: layers [0, 1, 2]
  Document 5: layers [0]
  Document 6: layers [0, 1, 2]
  Document 7: layers [0, 1]

Index statistics:
  Total vectors: 8
  Highest layer: 2

Testing search...
Top 3 search results:
  1. (similarity: 0.802) Natural language processing uses neural networks...
  2. (similarity: 0.798) Database systems store and retrieve information...
  3. (similarity: 0.790) Search algorithms find relevant information quickl...
```

## Performance Characteristics

- **Time Complexity**: O(log N) expected search time
- **Space Complexity**: O(N × M) where M is the average number of connections
- **Best For**: High-dimensional vectors (>50 dimensions)
- **Approximate**: Results are approximate but very fast

## Limitations & Future Improvements

- Currently uses random embeddings in demo (integrate with real embedding models)
- No persistence mechanism (could add save/load functionality)
- Single-threaded implementation (could benefit from parallelization)
- No dynamic deletion of vectors (could add removal functionality)

## License

This project is open source. Feel free to use and modify according to your needs.

## References

- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- Original HNSW paper by Yu. A. Malkov and D. A. Yashunin