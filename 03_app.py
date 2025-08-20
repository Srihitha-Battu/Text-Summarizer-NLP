# app.py

import streamlit as st
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import string

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuation = string.punctuation + '\n—“,”‘’”'

# --- Preprocessing ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    return text

def summarize_text(text, n=5):
    text = clean_text(text)
    words = word_tokenize(text.lower())
    freq_table = {}
    
    for word in words:
        if word not in stop_words and word not in punctuation:
            freq_table[word] = freq_table.get(word, 0) + 1

    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq_table[word]

    summary_sentences = nlargest(n, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def show_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# --- Load Sample Dataset ---
def load_sample():
    df1 = pd.read_csv("articles1.csv")
    df2 = pd.read_csv("articles2.csv")
    df3 = pd.read_csv("articles3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df.dropna(subset=['content'], inplace=True)
    df.rename(columns={'content': 'article'}, inplace=True)
    return df[['title', 'article']]


# --- Streamlit UI ---
st.set_page_config(page_title="📰 Advanced News Summarizer", layout="wide")
st.title("📰 Advanced News Article Summarizer")

tab1, tab2 = st.tabs(["📋 Summarize Custom Text", "📂 Explore Dataset"])

with tab1:
    st.subheader("Paste or type your article below")
    user_text = st.text_area("Enter news article", height=300)

    num_sentences = st.slider("Summary Length (in sentences)", 1, 10, 3)

    if st.button("Generate Summary"):
        if user_text.strip() == "":
            st.warning("Please enter a valid article.")
        else:
            summary = summarize_text(user_text, num_sentences)
            st.markdown("### ✂️ Summary")
            st.success(summary)

            st.markdown("### 🔍 Word Cloud")
            show_wordcloud(user_text)

            polarity, subjectivity = sentiment_analysis(user_text)
            st.markdown("### ❤️ Sentiment Analysis")
            st.info(f"Polarity: `{polarity:.2f}` | Subjectivity: `{subjectivity:.2f}`")

with tab2:
    st.subheader("Sample Dataset")
    df = load_sample()
    st.dataframe(df.head(20))

    idx = st.selectbox("Select an article to summarize", df.index[:20])
    article_text = df.loc[idx, 'article']
    st.markdown(f"**Title:** {df.loc[idx, 'title']}")
    st.markdown("#### Full Article:")
    st.write(article_text)

    summary_sample = summarize_text(article_text)
    st.markdown("#### ✂️ Summary:")
    st.success(summary_sample)

    st.markdown("#### 🔍 Word Cloud")
    show_wordcloud(article_text)

    polarity, subjectivity = sentiment_analysis(article_text)
    st.markdown("#### ❤️ Sentiment")
    st.info(f"Polarity: `{polarity:.2f}` | Subjectivity: `{subjectivity:.2f}`")
