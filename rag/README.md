# Bhagavad Gita RAG System for Kannada

This project implements a Retrieval Augmented Generation (RAG) system for the Bhagavad Gita in Kannada language. The system allows users to query the Bhagavad Gita in Kannada and retrieve the most relevant verses based on semantic similarity.

## Features

- Load and process Kannada JSON data of Bhagavad Gita
- Create embeddings for verses using a multilingual sentence transformer model
- Retrieve the most relevant verses for a given query using cosine similarity
- Support for various JSON structures of Bhagavad Gita data

## Requirements

```
numpy
scikit-learn
sentence-transformers
```

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install numpy scikit-learn sentence-transformers
```

## Usage

### Basic Usage

```python
from bhagavadgita_rag import BhagavadGitaRAG

# Initialize the RAG system with your Kannada Bhagavad Gita JSON file
json_path = "bhagavadgita_kannada_sample.json"
rag = BhagavadGitaRAG(json_path)

# Query in Kannada
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
```

### Sample Data

A sample JSON file with Kannada Bhagavad Gita verses is included in this repository (`bhagavadgita_kannada_sample.json`). This file contains selected verses from different chapters of the Bhagavad Gita in Kannada.

## How It Works

1. **Data Loading**: The system loads the Kannada JSON data of Bhagavad Gita and extracts verses from various possible JSON structures.

2. **Embedding Creation**: It uses a multilingual sentence transformer model to create embeddings for all verses. The default model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which supports Kannada language.

3. **Retrieval**: When a query is provided, the system creates an embedding for the query and calculates the cosine similarity between the query embedding and all verse embeddings. It then returns the top-k most similar verses.

## Customization

### Using a Different Embedding Model

You can use a different sentence transformer model by specifying it during initialization:

```python
rag = BhagavadGitaRAG(json_path, model_name="sentence-transformers/LaBSE")
```

### Adjusting the Number of Results

You can adjust the number of results returned by the `retrieve` method:

```python
results = rag.retrieve(query, top_k=5)  # Return top 5 results
```

## License

This project is open-source and available for educational and research purposes.