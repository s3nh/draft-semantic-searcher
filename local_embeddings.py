from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

class LocalSemanticMatcher:
    def __init__(self, knowledge_base, model_name='all-MiniLM-L6-v2'):
        """
        Lightweight models (choose based on your needs):
        - 'all-MiniLM-L6-v2'      → Fast, good quality (80MB)
        - 'all-mpnet-base-v2'     → Best quality (420MB)
        - 'paraphrase-MiniLM-L3-v2' → Fastest, smaller (60MB)
        - 'multi-qa-MiniLM-L6-cos-v1' → Optimized for Q&A
        """
        self.kb = knowledge_base
        self.model = SentenceTransformer(model_name)
        self.kb_embeddings = self._embed_knowledge_base()
    
    def _format_complaint(self, item):
        return (
            f"Channel: {item['channel']} | Group: {item['group']} | "
            f"Type: {item['type']} | Complaint: {item['content']}"
        )
    
    def _embed_knowledge_base(self):
        """Pre-compute and cache embeddings for entire KB"""
        texts = [self._format_complaint(item) for item in self.kb]
        # This runs locally - no API calls! 
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For faster cosine sim
        )
        return embeddings
    
    def save_embeddings(self, path='kb_embeddings.pkl'):
        """Cache embeddings to avoid recomputing"""
        with open(path, 'wb') as f:
            pickle.dump(self.kb_embeddings, f)
    
    def load_embeddings(self, path='kb_embeddings.pkl'):
        """Load cached embeddings"""
        with open(path, 'rb') as f:
            self.kb_embeddings = pickle. load(f)
    
    def find_best_answer(self, complaint, top_k=5, metadata_boost=0.2):
        # Embed the query
        query_text = self._format_complaint(complaint)
        query_embedding = self.model.encode(
            query_text, 
            normalize_embeddings=True
        )
        
        # Fast cosine similarity (dot product since normalized)
        semantic_scores = np.dot(self.kb_embeddings, query_embedding)
        
        # Optional: Boost for metadata matches
        metadata_scores = np.array([
            sum([
                complaint. get('channel') == item.get('channel'),
                complaint.get('group') == item.get('group'),
                complaint.get('type') == item.get('type')
            ]) / 3.0
            for item in self.kb
        ])
        
        final_scores = semantic_scores + (metadata_boost * metadata_scores)
        
        # Get top-k indices
        top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        
        return [
            {
                'answer': self.kb[i]['answer'],
                'original_complaint': self.kb[i]['content'],
                'semantic_score': float(semantic_scores[i]),
                'metadata_score': float(metadata_scores[i]),
                'final_score': float(final_scores[i]),
                'metadata':  {
                    'channel': self.kb[i]['channel'],
                    'group': self.kb[i]['group'],
                    'type': self.kb[i]['type']
                }
            }
            for i in top_indices
        ]


# Usage
knowledge_base = [
    {
        'channel': 'mobile',
        'group': 'transactions',
        'type': 'failed_payment',
        'content': 'My payment to vendor XYZ failed but money was deducted',
        'answer': 'We apologize for the inconvenience. Failed transactions are automatically reversed within 5-7 business days.. .'
    },
    # ... more entries
]

matcher = LocalSemanticMatcher(knowledge_base)

# Save embeddings for later (one-time computation)
matcher.save_embeddings('complaints_embeddings.pkl')

# Find answer for new complaint
new_complaint = {
    'channel': 'mobile',
    'group': 'transactions',
    'type': 'failed_payment',
    'content': 'I tried to pay my electricity bill but the transaction failed and amount is debited'
}

results = matcher.find_best_answer(new_complaint, top_k=3)
