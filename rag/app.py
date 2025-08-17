import streamlit as st
import os
from bhagavadgita_rag import BhagavadGitaRAG
from gtts import gTTS
import tempfile
import base64

# Set page config
st.set_page_config(
    page_title="ಭಗವದ್ಗೀತೆ ಸರ್ಚ್ | Bhagavad Gita Search",
    page_icon="🕉️",
    layout="wide"
)

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Language selection radio buttons
language = st.radio(
    "Select Language / ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ:",
    options=['English', 'Kannada'],
    index=0 if st.session_state.language == 'English' else 1,
    horizontal=True
)

# Update session state
st.session_state.language = language

# Language-specific content dictionary
LANG_CONTENT = {
    'English': {
        'title': '🕉️ Bhagavad Gita Search',
        'subtitle': 'Semantic Search in Bhagavad Gita',
        'data_loaded': '✅ Bhagavad Gita data loaded successfully',
        'example_queries_title': '📝 Example Queries',
        'example_queries': [
            'What is the true difference between Tyaga and Sannyasa?',
            'How do the three Gunas influence our actions and knowledge?',
            "What is the importance of performing one's own duty (Svadharma)?",
            'How can one attain liberation (Moksha) from the bondage of karma?',
            'What is the final advice given by Sri Krishna to Arjuna?',
        ],
        'query_placeholder': 'Enter your question here',
        'query_example': 'Example: What did Krishna say about karma?',
        'results_slider': 'How many results to show?',
        'search_button': '🔍 Search',
        'searching': 'Searching...',
        'results_title': '📖 Results',
        'chapter': 'Chapter',
        'verse': 'Verse',
        'similarity': 'Similarity',
        'original_verse': 'Original Verse',
        'translation': 'English Translation',
        'generating_audio': 'Generating audio...',
        'generating_original_audio': 'Generating original verse audio...',
        'generating_translation_audio': 'Generating translation audio...'
    },
    'Kannada': {
        'title': '🕉️ ಭಗವದ್ಗೀತೆ ಸರ್ಚ್',
        'subtitle': 'ಭಗವದ್ಗೀತೆಯಲ್ಲಿ ಸಂದರ್ಭೋಚಿತ ಹುಡುಕಾಟ',
        'data_loaded': '✅ ಭಗವದ್ಗೀತೆಯ ಮಾಹಿತಿ ಈಗ ಲಭ್ಯ',
        'example_queries_title': '📝 ಉದಾಹರಣೆ ಪ್ರಶ್ನೆಗಳು',
        'example_queries': [
            'ತ್ಯಾಗ ಮತ್ತು ಸಂನ್ಯಾಸದ ನಡುವಿನ ನಿಜವಾದ ವ್ಯತ್ಯಾಸವೇನು?', # What is the true difference between Tyaga and Sannyasa?
            'ಮೂರು ಗುಣಗಳು ನಮ್ಮ ಕರ್ಮ ಮತ್ತು ಜ್ಞಾನದ ಮೇಲೆ ಹೇಗೆ ಪ್ರಭಾವ ಬೀರುತ್ತವೆ?', # How do the three Gunas influence our actions and knowledge?
            'ಸ್ವಧರ್ಮವನ್ನು ಆಚರಿಸುವುದರ ಮಹತ್ವವೇನು?', # What is the importance of performing one's own duty (Svadharma)?
            'ಕರ್ಮ ಬಂಧನದಿಂದ ಮುಕ್ತರಾಗಿ ಮೋಕ್ಷವನ್ನು ಸಾಧಿಸುವುದು ಹೇಗೆ?', # How can one attain liberation (Moksha) from the bondage of karma?
            'ಅರ್ಜುನನಿಗೆ ಶ್ರೀಕೃಷ್ಣನು ನೀಡಿದ ಅಂತಿಮ ಉಪದೇಶವೇನು?', # What is the final advice given by Sri Krishna to Arjuna?
        ],
        'query_placeholder': 'ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ಬರೆಯಿರಿ',
        'query_example': 'ಉದಾ: ಕರ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣನು ಏನು ಹೇಳಿದನು?',
        'results_slider': 'ಎಷ್ಟು ಫಲಿತಾಂಶಗಳನ್ನು ತೋರಿಸಬೇಕು?',
        'search_button': '🔍 ಹುಡುಕಿ',
        'searching': 'ಹುಡುಕುತ್ತಿದೆ...',
        'results_title': '📖 ಫಲಿತಾಂಶಗಳು',
        'chapter': 'ಅಧ್ಯಾಯ',
        'verse': 'ಶ್ಲೋಕ',
        'similarity': 'ಹೊಂದಾಣಿಕೆ',
        'original_verse': 'ಮೂಲ ಶ್ಲೋಕ',
        'translation': 'ಕನ್ನಡ ಅರ್ಥ',
        'generating_audio': 'ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...',
        'generating_original_audio': 'ಮೂಲ ಶ್ಲೋಕದ ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...',
        'generating_translation_audio': 'ಅನುವಾದದ ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...'
    }
}

# Get current language content
content = LANG_CONTENT[st.session_state.language]

# Function to create audio player HTML
def get_audio_player_html(audio_path, label=""):
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_file.close()
    return f'''
    <div class="audio-section">
        <span class="audio-icon">🔊</span>
        <div class="audio-label">{label}</div>
        <div class="audio-player">
            <audio controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        </div>
    </div>
    '''

# Function to generate speech
def generate_speech(text, lang='kn'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stMarkdown {
        text-align: center;
    }
    .audio-section {
        display: flex;
        align-items: center;
        gap: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .audio-icon {
        font-size: 1.5em;
        color: #1f77b4;
    }
    .audio-label {
        font-weight: bold;
        color: #1f77b4;
        min-width: 120px;
    }
    .audio-player {
        flex-grow: 1;
    }
    .audio-player audio {
        width: 100%;
        margin: 5px 0;
    }
    .audio-player audio::-webkit-media-controls-panel {
        background-color: #ffffff;
    }
    .audio-player audio::-webkit-media-controls-play-button {
        background-color: #1f77b4;
        border-radius: 50%;
    }
    .audio-player audio::-webkit-media-controls-timeline {
        background-color: #e6e9ef;
        border-radius: 25px;
        margin: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title(content['title'])
st.markdown(f"""
### {content['subtitle']}
---
""")

@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "bhagavadgita_Chapter_18.json")
    return BhagavadGitaRAG(json_path)

# Initialize session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Initialize the RAG system
try:
    rag = load_rag_system()
    st.success(content['data_loaded'])

    # Example queries in sidebar
    with st.sidebar:
        st.markdown(f"### {content['example_queries_title']}")
        for example_query in content['example_queries']:
            if st.button(example_query):
                st.session_state.query = example_query

    # Search interface
    with st.form(key="search_form"):
        # Search query input
        query = st.text_input(
            content['query_placeholder'],
            value=st.session_state.query,
            placeholder=content['query_example']
        )

        # Number of results slider
        num_results = st.slider(
            content['results_slider'],
            min_value=1,
            max_value=10,
            value=3
        )

        # Search button
        search_button = st.form_submit_button(content['search_button'])

    # Process search when button is clicked or when example query is selected
    if (search_button and query) or (st.session_state.query and not search_button):
        # Use the query from session state if it exists, otherwise use the form input
        search_query = st.session_state.query if st.session_state.query else query
        
        with st.spinner(content['searching']):
            results = rag.retrieve(search_query, top_k=num_results)

        # Display results
        st.markdown(f"### {content['results_title']}")
        for i, result in enumerate(results, 1):
            verse = result['verse']
            similarity = result['similarity']

            with st.expander(f"{content['chapter']} {verse.get('chapter', 'Unknown')}, {content['verse']} {verse.get('verse', 'Unknown')} ({content['similarity']}: {similarity:.2%})"):
                # Original text
                st.markdown(f"**{content['original_verse']}:**")
                st.markdown(f"*{verse.get('text', '')}*")

                # Add text-to-speech for original verse
                with st.spinner(content['generating_original_audio']):
                    original_audio_path = generate_speech(verse.get('text', ''))
                    if original_audio_path:
                        st.markdown(get_audio_player_html(original_audio_path, content['original_verse']), unsafe_allow_html=True)
                        try:
                            os.unlink(original_audio_path)
                        except:
                            pass

                # Translation
                if 'translation' in verse:
                    st.markdown(f"**{content['translation']}:**")
                    
                    if st.session_state.language == 'English':
                        # For English mode, show English translation if available, otherwise show Kannada
                        english_translation = verse.get('english_translation', verse['translation'])
                        st.markdown(english_translation)
                        
                        # Add text-to-speech for English translation
                        with st.spinner(content['generating_translation_audio']):
                            translation_audio_path = generate_speech(english_translation, lang='en')
                            if translation_audio_path:
                                st.markdown(get_audio_player_html(translation_audio_path, content['translation']), unsafe_allow_html=True)
                                try:
                                    os.unlink(translation_audio_path)
                                except:
                                    pass
                    else:
                        # For Kannada mode, show Kannada translation
                        st.markdown(verse['translation'])
                        
                        # Add text-to-speech for Kannada translation
                        with st.spinner(content['generating_translation_audio']):
                            translation_audio_path = generate_speech(verse['translation'])
                            if translation_audio_path:
                                st.markdown(get_audio_player_html(translation_audio_path, content['translation']), unsafe_allow_html=True)
                                try:
                                    os.unlink(translation_audio_path)
                                except:
                                    pass

                # Similarity score
                st.progress(similarity)
        
        # Clear the session state query after search
        st.session_state.query = ""

except Exception as e:
    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)