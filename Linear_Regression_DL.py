import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Setting up device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in use: {device}\n")

# Lets get our data ready
weights = 0.3 
bias = 0.9
# Our data will be of these combinations (0.3 * X + 0.9)

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weights * X + bias
print(f"Number of X samples: {len(X)}")
print(f"Number of y samples: {len(y)}")
print("First 10 X & y samples:")
print(f"X: {X[:10]}")
print(f"y: {y[:10]}")


train_split = int(0.8 * len(X))
X_train = X[:train_split]
y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]

print(f"Train split: {len(X_train)}")
print(f"Test Split: {len(X_test)}")

def plot_predictions(train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None):
    """
    A helper function to visualize our data.

    Args:
        - predictions: Our Model predicted data 
    Returns:
        - A scatter plot visualizing the data.
    """
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_labels, c='b', s = 4, label = "Training Data")
    plt.scatter(test_data, test_labels, c='g', s = 4, label = "Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c = 'r', s = 4, label="Predicted Data")
    plt.legend()
    plt.show()

plot_predictions()

# Building our NN Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Linear layer for creating model parameters this is also referred as linear transform, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model = LinearRegressionModel()

print(f"Models is currently in: {next(model.parameters()).device}")
# Lets move it to GPU for faster training

model.to(device)
print(f"Models is changed to: {next(model.parameters()).device}")

## Training
loss_fn = nn.L1Loss() # (MAE)
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01)

torch.manual_seed(42)

# Putting data in our GPU

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

epochs = 300

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

print(model.state_dict())

model.eval()
with torch.inference_mode():
    y_preds = model(X_test)


plot_predictions(predictions=y_preds.cpu())
