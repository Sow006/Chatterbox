import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer

lemmatizer = WordNetLemmatizer()

# Sample chatbot corpus with more diverse responses

corpus = [
    "Hello! How can I assist you today?",
    "I am an AI chatbot designed to help you.",
    "I specialize in answering questions about AI, Python, and machine learning.",
    "My knowledge covers various topics in technology and programming.",
    "Feel free to ask me anything within my expertise.",
    "I'm here to provide informative and helpful responses.",
    "Goodbye! I hope I could be of help.",
    "Thank you for chatting with me today!"
]

# Preprocessing function

def preprocess_text(text):

    # Convert to lowercase
    
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)

def chatbot_response(user_input):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    # Create a copy of corpus and add processed user input
    sent_tokens = [preprocess_text(sent) for sent in corpus]
    sent_tokens.append(processed_input)
    
    # Vectorize the tokens
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sent_tokens)
    
    # Calculate similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Find the most similar response
    index = similarity_scores.argsort()[0][-1]
    
    # Check if similarity is above a threshold
    if similarity_scores[0, index] > 0.1:
        return corpus[index]
    else:
        # Fallback responses for low similarity
        fallback_responses = [
            "I'm sorry, I didn't quite understand that.",
            "Could you rephrase your question?",
            "I'm not sure I can help with that specific query.",
            "Can you provide more context?"
        ]
        return random.choice(fallback_responses)

def chat():
    print("Chatbot: Hello! Ask me anything or type 'bye' to exit.")
    while True:
        try:
            user_input = input("You: ").lower()
            if user_input in ['bye', 'goodbye', 'exit']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            response = chatbot_response(user_input)
            print("Chatbot:", response)
        except Exception as e:
            print("Chatbot: Sorry, something went wrong. Let's try again.")

if __name__ == "__main__":
    chat()