# Phase 4 — Train Decision Controller
# Learns mapping: latent features → (steer, throttle, brake)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ---------------- CONFIG ----------------
csv_path = r"E:\AutonomousVehicle\results\latent_drive_data_20251102_155805.csv"

save_path = r"E:\AutonomousVehicle\results\controller_model.pth"
epochs = 25
batch_size = 64
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ----------------
print("📂 Loading data from:", csv_path)
df = pd.read_csv(csv_path)

# Split input (latent features) and output (controls)
X = df.drop(columns=["throttle", "steer", "brake"]).values
y = df[["steer", "throttle", "brake"]].values

print("🧮 Data shape:", X.shape, "→", y.shape)

# Normalize input for stable training
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val = map(lambda a: torch.tensor(a, dtype=torch.float32), (X_train, X_val))
y_train, y_val = map(lambda a: torch.tensor(a, dtype=torch.float32), (y_train, y_val))

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
val_ds = torch.utils.data.TensorDataset(X_val, y_val)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

# ---------------- MODEL ----------------
class DecisionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DecisionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()  # steer, throttle, brake in [-1,1] range
        )

    def forward(self, x):
        return self.net(x)

model = DecisionModel(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("✅ Model initialized on", device)

# ---------------- TRAINING LOOP ----------------
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = np.mean([
            criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_dl
        ])

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_dl):.5f} | Val Loss: {val_loss:.5f}")

# ---------------- SAVE MODEL ----------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(model.state_dict(), save_path)
print(f"💾 Controller model saved ({timestamp}) → {save_path}")

# ---------------- SUMMARY ----------------
print("\n✅ Training complete!")
print("You can now integrate this model into CARLA for autonomous driving.")
print("Next: use human_like_control.py with this trained controller.")

