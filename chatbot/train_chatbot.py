# train_chatbot.py
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Initialize stemmer
# -----------------------------
stemmer = PorterStemmer()

# -----------------------------
# Preprocessing functions
# -----------------------------
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

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

# -----------------------------
# Create training data
# -----------------------------
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# -----------------------------
# Neural Network Definition
# -----------------------------
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

input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)

model = ChatNet(input_size, hidden_size, output_size)

# -----------------------------
# Training Setup
# -----------------------------
dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(1000):
    for words, labels in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/1000, Loss: {loss.item():.4f}')

# -----------------------------
# Save Trained Model
# -----------------------------
torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}, "chatbot_model.pth")

print("Training complete. Model saved as chatbot_model.pth")
