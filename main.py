import random
import numpy as np

class HNSW:
    def __init__(self, max_connections=16):
        self.max_connections = max_connections
        self.vec2neighbors = {}
        self.vec2embeds = {}
        self.vec2docs = {}
        self.highest_layer = 0
        self.currentID = 0

    def _assign_layer(self):
        """Probabilistic layer assignment using coin flips"""
        layer = 0
        while random.random() > 0.5:
            layer += 1
        return layer

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query_embedding, num_candidates=50):
        """Search for similar vectors using multi-layer beam search"""
        if len(self.vec2neighbors) == 0:
            return {}

        # Find entry points in top layer
        current_candidates = []
        for vector_id in self.vec2neighbors:
            if self.highest_layer in self.vec2neighbors[vector_id]:
                current_candidates.append(vector_id)

        layer_candidates = {}

        # Search down through layers
        for layer in range(self.highest_layer, -1, -1):
            # Collect neighbors of current candidates
            all_neighbors = []
            for candidate in current_candidates:
                if layer in self.vec2neighbors[candidate]:
                    neighbors = self.vec2neighbors[candidate][layer]
                    all_neighbors.extend(neighbors)

            # Combine candidates and neighbors
            all_candidates = current_candidates + all_neighbors
            unique_candidates = list(set(all_candidates))

            if not unique_candidates:
                layer_candidates[layer] = []
                current_candidates = []
                continue

            # Score by similarity
            scored_candidates = []
            for candidate in unique_candidates:
                similarity = self.cosine_similarity(query_embedding, self.vec2embeds[candidate])
                scored_candidates.append((similarity, candidate))

            # Keep top candidates
            scored_candidates.sort(reverse=True)
            top_candidates = scored_candidates[:num_candidates]

            layer_candidates[layer] = [vector_id for _, vector_id in top_candidates]

            # Prepare for next layer
            next_size = min(num_candidates // 2, len(top_candidates))
            current_candidates = [vector_id for _, vector_id in top_candidates[:next_size]]

        return layer_candidates

    def _make_connections(self, new_vector_id, candidates_by_layer, new_vector_layers):
        """Create bidirectional connections with connection limits"""
        for layer in new_vector_layers:
            if layer not in candidates_by_layer:
                continue

            candidates = candidates_by_layer[layer]
            connections_to_make = min(len(candidates), self.max_connections)
            best_candidates = candidates[:connections_to_make]

            for candidate_id in best_candidates:
                # Add forward connection
                self.vec2neighbors[new_vector_id][layer].append(candidate_id)

                # Add reverse connection with eviction logic
                if candidate_id in self.vec2neighbors and layer in self.vec2neighbors[candidate_id]:
                    candidate_neighbors = self.vec2neighbors[candidate_id][layer]

                    if len(candidate_neighbors) < self.max_connections:
                        candidate_neighbors.append(new_vector_id)
                    else:
                        # Need to evict worst neighbor
                        candidate_embedding = self.vec2embeds[candidate_id]
                        new_embedding = self.vec2embeds[new_vector_id]
                        new_similarity = self.cosine_similarity(candidate_embedding, new_embedding)

                        # Find worst existing neighbor
                        worst_similarity = float('inf')
                        worst_neighbor = None

                        for neighbor_id in candidate_neighbors:
                            neighbor_embedding = self.vec2embeds[neighbor_id]
                            similarity = self.cosine_similarity(candidate_embedding, neighbor_embedding)
                            if similarity < worst_similarity:
                                worst_similarity = similarity
                                worst_neighbor = neighbor_id

                        # Replace if new is better
                        if new_similarity > worst_similarity:
                            candidate_neighbors.remove(worst_neighbor)
                            candidate_neighbors.append(new_vector_id)

    def insert(self, embedding, document):
        """Insert a new vector into the index"""
        # Assign to layers
        max_layer = self._assign_layer()
        self.highest_layer = max(self.highest_layer, max_layer)
        vector_layers = list(range(max_layer + 1))

        # Find neighbors using search
        candidates_by_layer = {}
        if self.currentID > 0:
            candidates_by_layer = self.search(embedding, num_candidates=50)

        # Store vector data
        vector_id = self.currentID
        self.vec2embeds[vector_id] = embedding
        self.vec2docs[vector_id] = document

        # Initialize neighbor lists
        self.vec2neighbors[vector_id] = {}
        for layer in vector_layers:
            self.vec2neighbors[vector_id][layer] = []

        # Make connections
        if candidates_by_layer:
            self._make_connections(vector_id, candidates_by_layer, vector_layers)

        self.currentID += 1
        return vector_id

    def query(self, query_embedding, k=5):
        """User-friendly search returning top-k documents"""
        if len(self.vec2neighbors) == 0:
            return []

        layer_candidates = self.search(query_embedding, num_candidates=k*2)

        if 0 not in layer_candidates:
            return []

        candidates = layer_candidates[0]
        scored_results = []

        for candidate_id in candidates:
            similarity = self.cosine_similarity(query_embedding, self.vec2embeds[candidate_id])
            scored_results.append({
                'document': self.vec2docs[candidate_id],
                'similarity': similarity,
                'vector_id': candidate_id
            })

        scored_results.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_results[:k]

# Demo and Testing
def demo_hnsw():
    """Demonstrate our HNSW implementation"""
    print("Building HNSW Vector Database...")
    index = HNSW(max_connections=16)

    # Sample documents with simple embeddings
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Vector databases enable semantic search",
        "Python is a versatile programming language",
        "Natural language processing uses neural networks",
        "Artificial intelligence will change the world",
        "Database systems store and retrieve information",
        "Search algorithms find relevant information quickly"
    ]

    # Create simple test embeddings (in real use, get these from OpenAI, etc.)
    embeddings = []
    for i, doc in enumerate(documents):
        # Simple embedding based on document characteristics
        embedding = np.random.rand(64)  # 64-dimensional random vectors
        embeddings.append(embedding)

    # Insert all documents
    print(f"Inserting {len(documents)} documents...")
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        vector_id = index.insert(embedding, doc)
        layer_info = list(index.vec2neighbors[vector_id].keys())
        print(f"  Document {vector_id}: layers {layer_info}")

    print(f"\nIndex statistics:")
    print(f"  Total vectors: {len(index.vec2neighbors)}")
    print(f"  Highest layer: {index.highest_layer}")

    # Test search
    print(f"\nTesting search...")
    query_embedding = np.random.rand(64)
    results = index.query(query_embedding, k=3)

    print("Top 3 search results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. (similarity: {result['similarity']:.3f}) {result['document'][:50]}...")

if __name__ == "__main__":
    demo_hnsw()
