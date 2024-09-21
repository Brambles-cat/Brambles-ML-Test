import torch
from time import sleep
import time
import data_fetcher

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

epochs = 10
batch_size = 100
learning_rate = 0.03

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

train_set, test_set = data_fetcher.pull_data().values()
print("Datasets retrieved...")

train_set = TensorDataset(
    torch.tensor([example['image'] for example in train_set], device=device),
    torch.tensor([example['label'] for example in train_set], device=device)
)

test_features = torch.tensor([example['image'] for example in test_set], device=device)
test_labels = torch.tensor([example['label'] for example in test_set], device=device)
print("Testing data ready...")

loss_fn = nn.CrossEntropyLoss()

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.Sigmoid(),
    nn.Linear(100, 10),
    nn.Softmax(dim=1)
).to(device)

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

training_batches = DataLoader(train_set, batch_size=batch_size, shuffle=True)
print("Training data ready...")

model.train()

start = time.time()

for epoch in range(epochs):
    print(f"Epoch: {epoch}")

    for batch in training_batches:
        feature_batch, label_batch = batch

        test_outputs = model(feature_batch)
        loss = loss_fn(test_outputs, label_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

torch.cuda.synchronize()

print(time.time() - start)

model.eval()

with torch.inference_mode():
    train_outputs = model(train_set.tensors[0])
    train_preds = torch.argmax(train_outputs, dim=1)
    train_accuracy = (train_preds == train_set.tensors[1]).sum().item() / train_preds.size(0)
    print(f"Train Accuracy : %{train_accuracy * 100}")

    test_outputs = model(test_features)
    test_preds = torch.argmax(test_outputs, dim=1)
    test_accuracy = (test_preds == test_labels).sum().item() / test_preds.size(0)
    print(f"Test Accuracy  : %{test_accuracy * 100}")

