import faiss
import numpy as np
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer

# Store index and metadata lightly
INDEX_PATH = os.path.join(os.path.dirname(__file__), "rag_index.faiss")
META_PATH = os.path.join(os.path.dirname(__file__), "rag_meta.pkl")

class RAGMemory:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Loading Embedding Model (all-MiniLM-L6-v2) on {self.device.upper()}...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.dimension = 384
        
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            print("--> Loading existing RAG Index...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print("--> Creating new RAG Index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = [] # List of dicts: {"user_id": ..., "text": ..., "id": ...}
            
    def add_memory(self, user_id, text, interaction_id):
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding).astype('float32'))
        
        self.metadata.append({
            "user_id": user_id,
            "text": text,
            "interaction_id": interaction_id
        })
        self._save()
        
    def search(self, query_text, user_id=None, k=5):
        embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(embedding).astype('float32'), k * 5) # Search more to filter
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            if idx >= len(self.metadata): continue
            
            meta = self.metadata[idx]
            
            # Filter by user if requested
            if user_id and meta["user_id"] != user_id:
                continue
                
            results.append({
                "text": meta["text"],
                "score": float(distances[0][i]),
                "user_id": meta["user_id"]
            })
            
            if len(results) >= k:
                break
                
        return results
        
    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

if __name__ == "__main__":
    # Test
    rag = RAGMemory()
    rag.add_memory("u1", "I love blue shirts", 1)
    rag.add_memory("u1", "I hate wool", 2)
    rag.add_memory("u2", "I buy aggressive sneakers", 3)
    
    print(rag.search("what fabric do I dislike?", user_id="u1"))
