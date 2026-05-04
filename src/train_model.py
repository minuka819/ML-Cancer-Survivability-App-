import torch #core library (tensors = like NumPy arrays but for ML)
import torch.nn as nn #nn = neural network components
import torch.optim as optim #optimizers (how the model learns)
from sklearn.model_selection import train_test_split #splits data into training/testing
from sklearn.preprocessing import StandardScaler #normalizes features 
import pandas as pd #handles datasets 

# Load data
df = pd.read_csv("../data/raw/METABRIC_RNA_Mutation.csv", low_memory=False)

df = df.dropna(subset=["overall_survival"]) #removes rows where label is missing 
df["overall_survival"] = df["overall_survival"].astype(int)#the overall survival label is converted to type int

numeric_df = df.select_dtypes(include=["int64", "float64"])

X = numeric_df.drop(columns=["overall_survival", "patient_id"], errors="ignore")
#x - features , we remove overall survival and patient id 
y = df["overall_survival"]
# y = labels this is what the model tries to predict 
X = X.fillna(X.median())
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#setting the test size to 20% and training based on 80%

# Scale features
scaler = StandardScaler() #creates a scalar object
X_train = scaler.fit_transform(X_train) #learns mean/std from training data and scales it
X_test = scaler.transform(X_test) #applies same scaling to test data

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32) #converts NumPy -> PyTorch tensor
X_test = torch.tensor(X_test, dtype=torch.float32) #converts labels 
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Model
class CancerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = CancerModel(X_train.shape[1])

# Loss + optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model.eval()
with torch.no_grad():
    probs = model(X_test)  # raw probabilities (0–1)
    preds = (probs > 0.5).float()  # convert to 0/1

# Accuracy
accuracy = (preds == y_test).float().mean()
print("\nAccuracy:", accuracy.item())

# Convert tensors → numpy for sklearn
y_true = y_test.numpy()
y_pred = preds.numpy()
y_prob = probs.numpy()

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ROC-AUC
print("\nROC-AUC Score:")
print(roc_auc_score(y_true, y_prob))

print("\nTest Accuracy:", accuracy.item())