import torch
import torch.nn as nn
import torch.optim as optim


# Create training data
inputs = torch.tensor(
    [
        [22, 25], [25, 35], [47, 80], [52, 95], [46, 82], [56, 90],
        [23, 27], [30, 50], [40, 60], [39, 57], [53, 95], [48, 88
                                                           ]],
    dtype=torch.float32)
labels = torch.tensor([
    [0], [0], [1], [1], [1], [1], [0], [1], [1], [0], [1], [1
                                                            ]], dtype=torch.float32)

# Init a model with 1 layer, activation Sigmoid, SGD, BCELoss
model = nn.Sequential(
    nn.Linear(inputs.shape[1], 1),
    nn.Sigmoid()
)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # binary cross-entropy loss

# Train for 500 epochs
for step in range(500):
    # Clear old gradients
    optimizer.zero_grad()
    # Calculate loss function
    loss = criterion(model(inputs), labels)
    # Backpropagation
    loss.backward()
    # Update weight, bias
    optimizer.step()