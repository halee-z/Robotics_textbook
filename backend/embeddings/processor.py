import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Process and manage embeddings for the educational AI & humanoid robotics platform
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding processor
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the embedding model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning(
                f"Could not import sentence_transformers. "
                f"Install with: pip install sentence-transformers"
            )
            # Fallback to dummy implementation
            self.model = "dummy_model"
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into an embedding vector
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model == "dummy_model":
            # For the dummy model, return a fixed-size random vector
            return np.random.rand(384).astype(np.float32)
        else:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].astype(np.float32)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple text strings into embedding vectors
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            Embedding vectors as numpy array (batch_size, embedding_dim)
        """
        if self.model == "dummy_model":
            # For dummy model, return random vectors
            return np.random.rand(len(texts), 384).astype(np.float32)
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_closest_match(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray) -> int:
        """
        Find the index of the closest matching embedding
        
        Args:
            query_embedding: Embedding to match against
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Index of the closest match
        """
        similarities = []
        for candidate in candidate_embeddings:
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append(sim)
        
        # Find index with maximum similarity
        max_idx = np.argmax(similarities)
        return int(max_idx)
    
    def create_document_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create embeddings for a collection of documents
        
        Args:
            documents: List of documents with 'id', 'text', 'metadata' keys
            
        Returns:
            Dictionary with embeddings and document information
        """
        texts = [doc.get('text', '') for doc in documents]
        embeddings = self.encode_texts(texts)
        
        return {
            'document_ids': [doc.get('id', f'doc_{i}') for i, doc in enumerate(documents)],
            'embeddings': embeddings,
            'texts': texts,
            'metadata': [doc.get('metadata', {}) for doc in documents]
        }
    
    def batch_compute_similarities(self, query_embedding: np.ndarray, 
                                  candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query embedding and multiple candidates
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        )
        
        # Compute similarities using dot product
        similarities = np.dot(candidate_norms, query_norm)
        return similarities

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = EmbeddingProcessor()
    
    # Test encoding
    text = "This is a test sentence for embedding"
    embedding = processor.encode_text(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Sample of embedding: {embedding[:10]}")  # First 10 values
    
    # Test similarity
    text2 = "Another test sentence"
    embedding2 = processor.encode_text(text2)
    similarity = processor.compute_similarity(embedding, embedding2)
    print(f"Similarity between texts: {similarity:.4f}")