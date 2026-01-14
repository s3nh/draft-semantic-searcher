### Rule based 


def find_answer(new_complaint, knowledge_base):
    # Filter by exact metadata match first
    candidates = [
        kb for kb in knowledge_base
        if kb['channel'] == new_complaint['channel']
        and kb['group'] == new_complaint['group']
        and kb['type'] == new_complaint['type']
    ]
    
    if candidates:
        # Return the most recent or most common answer
        return candidates[0]['answer']
    return None


### Tfdif + cosine 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ComplaintMatcher:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Combine metadata + content for matching
        self.corpus = [
            f"{item['group']} {item['type']} {item['channel']} {item['content']}"
            for item in knowledge_base
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
    
    def find_best_answer(self, complaint, top_k=3):
        query = f"{complaint['group']} {complaint['type']} {complaint['channel']} {complaint['content']}"
        query_vec = self.vectorizer. transform([query])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            {'answer': self.kb[i]['answer'], 'score': similarities[i]}
            for i in top_indices
        ]




### Bm25 matching 

from rank_bm25 import BM25Okapi

class BM25Matcher:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        corpus = [
            f"{item['group']} {item['type']} {item['channel']} {item['content']}".split()
            for item in knowledge_base
        ]
        self. bm25 = BM25Okapi(corpus)
    
    def find_best_answer(self, complaint, top_k=3):
        query = f"{complaint['group']} {complaint['type']} {complaint['channel']} {complaint['content']}".split()
        scores = self.bm25.get_scores(query)
        top_indices = scores.argsort()[-top_k:][::-1]
        
        return [self.kb[i]['answer'] for i in top_indices]

  
