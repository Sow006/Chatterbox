Overview

This project is a simple command-line chatbot (chatterbox) built in Python. It uses basic natural language processing (NLP) techniques to respond to user input by finding the most similar response from a predefined set of sentences.

Features
Preprocesses user input (lowercase, punctuation removal, lemmatization)
Uses TF-IDF vectorization and cosine similarity to match user queries to responses
Provides fallback responses for unrecognized queries
Runs in the terminal/command prompt
Requirements
Python 3.x
nltk
numpy
scikit-learn

Install dependencies with:

bash
pip install nltk numpy scikit-learn
Setup
Clone or download this repository.

Download NLTK data (first run only):

python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
Or, the script will attempt to do this automatically.

Ensure your script (e.g., chatterbox.py) is in your working directory.

Usage
Run the chatbot from your terminal:

bash

python chatterbox.py

Sample interaction:

text
Chatbot: Hello! Ask me anything or type 'bye' to exit.
You: What do you know about python?
Chatbot: I specialize in answering questions about AI, Python, and machine learning.
You: bye
Chatbot: Goodbye! Have a great day!
Customization
Edit the corpus list in the script to add or change responses.
