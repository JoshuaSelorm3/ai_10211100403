# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


from embedding import TextEmbedder

class ElectionDataRetriever:
    def __init__(self, embedder=None):
        """Initialize with a TextEmbedder instance"""
        self.embedder = embedder if embedder else TextEmbedder()
        
    def setup_from_chunks(self, chunks):
        """Create embeddings and build vector store from chunks"""
        embeddings = self.embedder.create_embeddings(chunks)
        self.embedder.build_vector_store(embeddings)
        return self
    
    def setup_from_saved(self, directory="./vector_store"):
        """Load a previously saved vector store"""
        self.embedder.load_vector_store(directory)
        return self
        
    def retrieve(self, query, k=5):
        """Retrieve relevant chunks for a given query"""
        raw_results = self.embedder.search(query, k)
        
        # Extract text from results and preserve scores
        processed_results = []
        for item in raw_results:
            processed_results.append({
                "text": item["chunk"]["text"],
                "metadata": item["chunk"]["metadata"],
                "score": item["score"]
            })
            
        return processed_results
    
    def format_for_llm(self, retrieved_chunks):
        """Format retrieved chunks into context for the LLM"""
        context = "Here is the relevant information about Ghana election results:\n\n"
        
        # Sort by score for relevance
        retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Add each chunk with a separator
        for i, chunk in enumerate(retrieved_chunks):
            context += f"{i+1}. {chunk['text']}\n"
            
        return context