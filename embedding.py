# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model for embeddings"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = None
        
    def create_embeddings(self, chunks):
        """Generate embeddings for text chunks"""
        self.chunks = chunks
        
        # Extract just the text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        return embeddings
    
    def build_vector_store(self, embeddings):
        """Create a FAISS index for vector similarity search"""
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(normalized_embeddings.astype('float32'))
        
        return self.index
    
    def save_vector_store(self, directory="./vector_store"):
        """Save the FAISS index and chunks for later use"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the FAISS index
        faiss.write_index(self.index, f"{directory}/election_data.index")
        
        # Save the chunks that correspond to the index
        with open(f"{directory}/election_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
    def load_vector_store(self, directory="./vector_store"):
        """Load a previously saved FAISS index and chunks"""
        # Load the FAISS index
        self.index = faiss.read_index(f"{directory}/election_data.index")
        
        # Load the corresponding chunks
        with open(f"{directory}/election_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
            
        return self.index, self.chunks
    
    def search(self, query, k=5):
        """Search for similar chunks given a query"""
        if self.index is None:
            raise ValueError("Vector store not built or loaded. Build or load an index first.")
        
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return the most relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Ensure index is valid
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[0][i])
                })
        
        return results