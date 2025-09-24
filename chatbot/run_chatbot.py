# run_chatbot.py
import torch
import torch.nn as nn
import random
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Preprocessing functions
# -----------------------------
stemmer = PorterStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# -----------------------------
# Load intents
# -----------------------------
with open('intents.json', 'r') as f:
    intents = json.load(f)

# -----------------------------
# Load trained model
# -----------------------------
data = torch.load("chatbot_model.pth")

all_words = data['all_words']
tags = data['tags']

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']

# Neural network definition
class ChatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ChatNet(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()

# -----------------------------
# Chat function
# -----------------------------
def chat():
    print("Start chatting with the bot (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        
        # Preprocess input
        bag = bag_of_words(tokenize(sentence), all_words)
        bag = torch.from_numpy(bag).unsqueeze(0)
        
        # Model prediction
        output = model(bag)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        
        # Responses
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print("Bot:", random.choice(intent['responses']))

# Start chatting
chat()
