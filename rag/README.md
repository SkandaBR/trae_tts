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

3. Create a virtual environment:

py -m venv .venv

3. Install the required packages:

```bash
pip install -r requirements.txt
```

# Quick Start Guide

## Running the Application

1. **Navigate to the project directory:**
   ```bash
   cd trae_tts\rag
   ```

2. **Create and activate virtual environment (if not already done):**
   ```bash
   py -m venv .venv311
   .venv311\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   - Open your web browser
   - Navigate to: `http://localhost:8501`
   - The bilingual Bhagavad Gita RAG interface will load successfully

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

# Chroma DB Integration Summary
# - Purpose: Chroma DB provides a persistent vector store for semantic retrieval of Bhagavad Gita content, enabling efficient top‑k search across Kannada/English text.
# - build_chroma_store.py: Ingests Chapter 18 JSON, constructs multilingual documents and metadata, computes normalized embeddings (SentenceTransformer multilingual‑e5‑large), and persists to collection `bhagavadgita_ch18` in `chroma_db`.
# - example_chroma_query.py: Encodes a Kannada query with the same embedding model, queries `bhagavadgita_ch18`, and prints top‑k results with verse metadata and document excerpts.
# - health_check_chroma.py: Executes a diagnostic write/read by adding a 1024‑dim test vector to `health_check`, querying it, and printing collection count and IDs with telemetry disabled.
# - Technical achievements: Stable IDs for deduplication, multilingual concatenation of text and translations for richer semantics, normalized embeddings, and reproducible local persistence (DuckDB/Parquet via Chroma).
# - Performance improvements: Precomputed embeddings and local persistence reduce query latency; embedding normalization improves cosine‑similarity stability; local client avoids network overhead.
# - Challenges overcome: Robust handling of heterogeneous JSON schemas and bilingual content alignment; safe collection reset logic to prevent stale data without native truncate.
# - Embedding method: SentenceTransformer `intfloat/multilingual-e5-large`, generating 1024‑dim normalized vectors for documents and queries.
# - Query capabilities: `collection.query` with configurable `n_results` (e.g., 3–5), returning documents and metadatas suitable for UI display or downstream processing.
# - Integration points: Shared `chroma_db` path reused across scripts; artifacts are ready for consumption by applications (e.g., `app.py`) or CLI utilities.