import streamlit as st
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

@st.cache_data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', '', text)
    return text

@st.cache_data
def tokenize_document(doc):
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

@st.cache_data
def summarize_document(samp_doc):
    sentences = nltk.sent_tokenize(samp_doc)
    num_sentences = len(sentences)
    
    if num_sentences == 0:
        return "No sentences to summarize."
    
    # Calculate the number of sentences for the summary (1/3 of input)
    n_sent = max(1, num_sentences // 3)

    # Tokenize and create the document-term matrix
    norm_sentences = [tokenize_document(sentence) for sentence in sentences]
    
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences).toarray()

    if dt_matrix.shape[1] == 0:
        return "Error in creating document-term matrix."

    try:
        n_topic = min(5, dt_matrix.shape[1])
        u, s, vt = svds(dt_matrix, k=n_topic)
        
        # Calculate salience scores
        salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
        top_sentence_indices = (-salience_scores).argsort()[:n_sent]

        # Sort indices before using them
        top_sentence_indices = np.sort(top_sentence_indices)

        # Extract sentences based on indices
        summary_sentences = [sentences[i] for i in top_sentence_indices if i < num_sentences]
        
        if not summary_sentences:
            return "Error: No valid sentences selected for summary."

        return ' '.join(summary_sentences)
    
    except Exception as e:
        return str(e)  # Handle errors gracefully

# Streamlit UI
st.title("Text Summarizer")
st.write("Paste your text below:")

# Input box for user text
user_input = st.text_area("Text to summarize:", height=300, key="user_input")

if st.button("Summarize"):
    if user_input:
        summary = summarize_document(user_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
