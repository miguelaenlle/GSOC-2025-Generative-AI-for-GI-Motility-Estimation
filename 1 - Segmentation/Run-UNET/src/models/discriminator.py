import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Define the model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=3)   # padding=3 for 'same'
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # padding=2 for 'same'
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # padding=1 for 'same'
        self.bn3   = nn.BatchNorm2d(128)

        # GlobalAveragePooling → AdaptiveAvgPool2d(1x1), then flatten
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1   = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)             # shape: [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)   # shape: [batch, 128]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # output between 0 and 1
        return x

# Sample training loop
def train_discriminator(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_results = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)         # shape: [batch, 1, 266, 266]
            labels = labels.to(device).float() # shape: [batch], values 0 or 1

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1) # shape: [batch]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
    return pd.DataFrame(training_results)

    

# Example usage:
if __name__ == "__main__":
    # Dummy Dataset example; replace with your actual Dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            # Return a random 1×266×266 tensor and a random binary label
            img = torch.randn(1, 266, 266)
            label = torch.randint(0, 2, (1,)).item()
            return img, label

    batch_size = 16
    train_ds = DummyDataset(500)
    val_ds   = DummyDataset(100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier()
    train_discriminator(model, train_ds, val_loader, device, num_epochs=5, lr=1e-3)
