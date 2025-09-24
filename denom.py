# denom.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ✅ Paths
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "denomination_model.pth")

# ✅ DenominationNet definition
class DenominationNet(nn.Module):
    def __init__(self):
        super(DenominationNet, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # 6 outputs for 2000, 500, 200, 100, 50, 20

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Dummy training for testing (optional, you can extend)
def train_and_save_model():
    model = DenominationNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy dataset (amounts -> denominations)
    # Example: amount = 500, output might be [0, 1, 0, 0, 0, 0] (one 500 note)
    X = torch.tensor([[500.0], [2000.0], [1000.0], [1500.0]], dtype=torch.float32)
    y = torch.tensor([[0,1,0,0,0,0],
                      [1,0,0,0,0,0],
                      [0,2,0,0,0,0],
                      [0,3,0,0,0,0]], dtype=torch.float32)

    # Train for a few epochs
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model trained and saved at {model_path}")

if __name__ == "__main__":
    train_and_save_model()
