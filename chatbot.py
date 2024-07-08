import streamlit as st
from transformers import pipeline
import random

# Load the sentiment-analysis and conversational pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
chatbot_pipeline = pipeline("conversational", model="microsoft/DialoGPT-small")

# Helper function to generate a response based on sentiment
def generate_response(user_input):
    sentiment = sentiment_pipeline(user_input)[0]
    if sentiment['label'] == 'NEGATIVE':
        response = "I'm sorry to hear that you're feeling this way. Here are some tips that might help: " \
                   "1. Take deep breaths. " \
                   "2. Write down your thoughts. " \
                   "3. Talk to a friend or loved one."
    else:
        response = "That's great to hear! Keep up the positive attitude. Remember, maintaining a healthy lifestyle is key to staying happy."
    
    # Get a conversational response from the model
    bot_input = chatbot_pipeline(user_input)
    bot_response = bot_input[0]['generated_text']
    
    return f"{response}\n\nChatbot: {bot_response}"

# Streamlit UI
st.title("AI-Powered Mental Health Chatbot")
st.write("Welcome to the mental health chatbot. How can I assist you today?")

# Text input from user
user_input = st.text_input("You: ", "")

# When the user submits input, generate a response
if user_input:
    response = generate_response(user_input)
    st.text_area("Chatbot: ", value=response, height=200)
