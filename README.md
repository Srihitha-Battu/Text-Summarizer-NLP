# Text-Summarizer-NLP
An NLP-based tool that summarizes large blocks of text using frequency-based extractive methods, built with Python and Streamlit.

This project is a simple yet powerful text summarizer that leverages Natural Language Processing (NLP) to condense long-form text into concise summaries. It supports both manual text input and CSV file uploads. Built using Python, it includes a Jupyter Notebook for exploration and a Streamlit interface for real-time use. Ideal for students, content analysts, and developers looking to understand or apply extractive summarization techniques.

## 📄 Project Description

This NLP-based text summarizer reduces lengthy content into concise and meaningful summaries. It supports raw user input or CSV-uploaded text data. Built with Python, the system is suitable for quick analysis, educational use, and prototype deployments.

---

## 🚀 Features

- **Text Preprocessing**: Cleans, tokenizes, and normalizes input text.
- **Summarization Logic**: Uses frequency scoring to extract the most important sentences.
- **CSV Input Support**: Works with uploaded `.csv` files containing text data.
- **Interactive UI**: Streamlit interface for real-time summarization and visualization.
- **Jupyter Notebook**: For step-by-step development and exploration.

---

## 🧱 Folder Structure

Text-Summariser-NLP/
│
├── app.py # Streamlit interface
├── summarizer.py # Core summarization logic
├── text_summarizer.ipynb # Main Jupyter notebook
├── data/
│ ├── input1.csv
│ ├── input2.csv
│ └── input3.csv # Sample inputs
├── requirements.txt # All required libraries
└── README.md # You're here!

## ⚙️ How It Works

1. **Input**: User enters text manually or selects a CSV file.
2. **Preprocessing**: Removes special characters, lowercases, and tokenizes sentences.
3. **Frequency Scoring**: Calculates word importance and scores sentences.
4. **Summary Extraction**: Selects top-ranked sentences to form a summary.
5. **Display**: Shows output on the Streamlit interface or notebook.

---

## 🛠️ Installation

### ✅ Requirements
- Python 3.8+
- Libraries: Streamlit, NLTK, Pandas, NumPy, Matplotlib, Seaborn, TextBlob, Wordcloud

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
