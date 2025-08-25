import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BhagavadGitaRAG:
    """
    Retrieval Augmented Generation system for Bhagavad Gita in Kannada.
    This class loads Kannada JSON data of Bhagavad Gita, indexes it using embeddings,
    and provides retrieval capabilities.
    """
    
    def __init__(self, json_path: str, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the RAG system.
        
        Args:
            json_path: Path to the Kannada JSON file of Bhagavad Gita
            model_name: Name of the sentence transformer model to use for embeddings
        """
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)
        self.verses = []
        self.embeddings = None
        self.load_data()
        self.create_embeddings()
    
    def load_data(self) -> None:
        """
        Load the Bhagavad Gita data from JSON file.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process the data based on expected structure
            # This might need adjustment based on the actual JSON structure
            if isinstance(data, list):
                self.verses = data
            elif isinstance(data, dict) and 'verses' in data:
                self.verses = data['verses']
            else:
                # Try to extract verses from the structure
                self.verses = self._extract_verses(data)
                
            print(f"Loaded {len(self.verses)} verses from Bhagavad Gita")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file at {self.json_path}")
    
    def _extract_verses(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract verses from a nested JSON structure.
        
        Args:
            data: The JSON data
            
        Returns:
            A list of verse dictionaries
        """
        verses = []
        
        # Handle different possible structures
        if 'chapters' in data:
            for chapter in data['chapters']:
                if 'verses' in chapter:
                    for verse in chapter['verses']:
                        verse['chapter'] = chapter.get('chapter_number', '')
                        verses.append(verse)
        
        # If no verses found, try flattening the structure
        if not verses:
            for chapter_num, chapter_data in data.items():
                if isinstance(chapter_data, dict) and 'verses' in chapter_data:
                    for verse_num, verse_text in chapter_data['verses'].items():
                        verses.append({
                            'chapter': chapter_num,
                            'verse': verse_num,
                            'text': verse_text
                        })
        
        return verses
    
    def create_embeddings(self) -> None:
        """
        Create embeddings for all verses.
        """
        if not self.verses:
            print("No verses to embed")
            return
        
        # Extract text from verses for embedding
        texts = []
        for verse in self.verses:
            if isinstance(verse, dict) and 'text' in verse:
                texts.append(verse['text'])
            elif isinstance(verse, str):
                texts.append(verse)
            else:
                # Try to find text in the verse structure
                text = self._extract_text(verse)
                texts.append(text if text else str(verse))
        
        # Create embeddings
        self.embeddings = self.model.encode(texts)
        print(f"Created embeddings for {len(texts)} verses")
    
    def _extract_text(self, verse: Dict[str, Any]) -> str:
        """
        Extract text from a verse dictionary with unknown structure.
        
        Args:
            verse: A verse dictionary
            
        Returns:
            The extracted text
        """
        # Try common field names for text
        for field in ['text', 'verse_text', 'content', 'kannada', 'translation']:
            if field in verse:
                return verse[field]
        
        # If no text field found, convert the whole verse to string
        return str(verse)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant verses for the query.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            List of top_k most relevant verses with similarity scores
        """
        if not self.verses or self.embeddings is None:
            raise ValueError("No verses or embeddings available")
        
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top_k verses with scores
        results = []
        for idx in top_indices:
            results.append({
                'verse': self.verses[idx],
                'similarity': float(similarities[idx])
            })
        
        return results


def main():
    """
    Example usage of the BhagavadGitaRAG class.
    """
    # Path to your Kannada Bhagavad Gita JSON file
    json_path = "path/to/bhagavadgita_Chapter_18.json"
    
    # Initialize the RAG system
    try:
        rag = BhagavadGitaRAG(json_path)
        
        # Example query in Kannada
        query = "ಕರ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣನು ಏನು ಹೇಳಿದನು?"  # "What did Krishna say about karma?"
        
        # Retrieve relevant verses
        results = rag.retrieve(query, top_k=3)
        
        # Print results
        print(f"\nQuery: {query}")
        print("\nRelevant verses:")
        for i, result in enumerate(results, 1):
            verse = result['verse']
            similarity = result['similarity']
            
            # Extract verse information
            chapter = verse.get('chapter', 'Unknown')
            verse_num = verse.get('verse', 'Unknown')
            text = verse.get('text', str(verse))
            
            print(f"\n{i}. Chapter {chapter}, Verse {verse_num} (Similarity: {similarity:.4f})")
            print(f"   {text}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()