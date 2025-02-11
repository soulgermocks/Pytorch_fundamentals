import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
PLOTS_PATH = Path('plots')
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Setting up device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device in use: {device}\n")

# Lets get our data ready
# Our data will be of these combinations (0.3 * X + 0.9)
weights = 0.3 
bias = 0.9


start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weights * X + bias
print(f"Number of X samples: \n{len(X)}")
print(f"Number of y samples: \n{len(y)}")
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

def plot_predictions(training_status,                 
                    train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None,
    ):
    """
    A helper function to visualize our data.

    Args:
        - training_status: Save name for the plot that we're saving.
        - predictions: Our Model predicted data.
    Returns:
        - A scatter plot visualizing the data.
    """
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_labels, c='b', s = 4, label = "Training Data")
    plt.scatter(test_data, test_labels, c='g', s = 4, label = "Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c = 'r', s = 4, label="Predicted Data")
    plt.legend()
    #plt.show()
    plt.savefig(f"plots/{training_status}.png")

plot_predictions(training_status="before_training")

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

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_train, y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform Backpropogation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Testing 
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # Printing out train stats
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.3f} | Test loss: {test_loss:.3f}")

print(model.state_dict())

model.eval()
with torch.inference_mode():
    y_preds = model(X_test)


plot_predictions(training_status="after_training", predictions=y_preds.cpu())

# Saving the model

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'linear_regression_300.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Saving the models state_dict

print(f"Saving the model to {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)

# Load the model 

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load('models/linear_regression_300.pth'))
loaded_model.to(device)

print(f"The loaded model is on device: {next(loaded_model.parameters()).device}")

print(f"Parameters of the Original Model: {model.state_dict()}\n")
print(f"Parameters of the Loaded Model: {loaded_model.state_dict()}")

# Evaluate the loaded model

loaded_model.eval()
with torch.inference_mode():
    loaded_model_y_preds = loaded_model(X_test)
print(f"{y_preds == loaded_model_y_preds}")