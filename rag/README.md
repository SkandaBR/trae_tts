# Bhagavad Gita RAG System with Bilingual Interface

This project implements a Retrieval Augmented Generation (RAG) system for the Bhagavad Gita with support for both Kannada and English languages. The system includes both a Python library for programmatic access and a Streamlit web application with an intuitive bilingual interface.

## Features

### Core RAG System
- Load and process Kannada JSON data of Bhagavad Gita
- Create embeddings for verses using a multilingual sentence transformer model
- Retrieve the most relevant verses for a given query using cosine similarity
- Support for various JSON structures of Bhagavad Gita data

### Streamlit Web Application
- **Bilingual Interface**: Complete language switching between English and Kannada
- **Dynamic Content**: All UI elements update based on language selection
- **Audio Support**: Text-to-speech functionality in both languages using gTTS
- **Interactive Search**: Real-time verse retrieval with similarity scoring
- **Example Queries**: Pre-built questions about key concepts like Dharma, Karma, and Moksha
- **Responsive Design**: Modern UI with expandable result sections

## Requirements

```
numpy
scikit-learn
sentence-transformers
streamlit
gtts
```

## Installation

1. Clone this repository
2. Navigate to the rag directory:

```bash
cd rag
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Streamlit Application

To launch the web interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Application Features

- **Language Selection**: Use the radio buttons to switch between English and Kannada
- **Search Interface**: Enter your query in either language
- **Audio Playback**: Listen to verses and translations with built-in text-to-speech
- **Example Queries**: Click on pre-built questions to explore key concepts
- **Adjustable Results**: Use the slider to control the number of search results

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

## File Structure

```
rag/
├── app.py                              # Streamlit web application
├── bhagavadgita_rag.py                 # Core RAG system implementation
├── bhagavadgita_kannada_sample.json    # Sample Bhagavad Gita data with translations
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Language Support

The application supports:
- **Kannada**: Native language interface with complete translations
- **English**: Full English interface with translated content
- **Audio**: Text-to-speech in both languages using Google Text-to-Speech (gTTS)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the system.

## License

This project is open-source and available for educational and research purposes.