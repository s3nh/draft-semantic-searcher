from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class FAISSSemanticMatcher:
    def __init__(self, knowledge_base, model_name='all-MiniLM-L6-v2'):
        self.kb = knowledge_base
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Build FAISS index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        texts = [item['content'] for item in self. kb]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Choose index type based on KB size
        if len(self. kb) < 10000:
            # Exact search for smaller KBs
            self. index = faiss.IndexFlatIP(self.dimension)
        else:
            # Approximate search for larger KBs (faster)
            nlist = min(100, len(self.kb) // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
        
        self.index.add(embeddings)
    
    def save_index(self, index_path='faiss. index', kb_path='kb.json'):
        """Persist index and KB"""
        faiss.write_index(self. index, index_path)
        with open(kb_path, 'w') as f:
            json.dump(self.kb, f)
    
    def load_index(self, index_path='faiss.index', kb_path='kb.json'):
        """Load persisted index"""
        self.index = faiss.read_index(index_path)
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)
    
    def find_best_answer(self, complaint, top_k=5):
        # Encode query
        query_embedding = self. model.encode([complaint['content']])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            results.append({
                'answer':  self.kb[idx]['answer'],
                'original_complaint': self.kb[idx]['content'],
                'score': float(score),
                'metadata': {
                    'channel': self.kb[idx]['channel'],
                    'group': self.kb[idx]['group'],
                    'type':  self.kb[idx]['type']
                }
            })
        
        return results


# Usage with metadata post-filtering
class FAISSWithMetadataFilter(FAISSSemanticMatcher):
    def find_best_answer(self, complaint, top_k=5, fetch_k=50):
        """Fetch more results, then filter by metadata"""
        # Get more candidates than needed
        query_embedding = self.model.encode([complaint['content']])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        # Score with metadata boost
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            item = self.kb[idx]
            metadata_match = sum([
                complaint.get('channel') == item.get('channel'),
                complaint.get('group') == item.get('group'),
                complaint.get('type') == item.get('type')
            ])
            
            # Boost score based on metadata matches
            boosted_score = score + (metadata_match * 0.1)
            
            results.append({
                'answer': item['answer'],
                'original_complaint': item['content'],
                'semantic_score': float(score),
                'metadata_matches': metadata_match,
                'final_score': float(boosted_score),
                'metadata': {
                    'channel': item['channel'],
                    'group':  item['group'],
                    'type': item['type']
                }
            })
        
        # Sort by boosted score and return top_k
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]
