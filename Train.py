import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from Dataset import ScatterPlotDataset, image_transforms
from tqdm import tqdm
from torchvision.models import ResNet18_Weights

""" Split the dataset """
def Split_data(dataset, seed):
    total_count = len(dataset)
    train_count = int(0.8 * total_count)
    val_count = int(0.1 * total_count)
    test_count = total_count - train_count - val_count  # To ensure the total is correct
    train_set, val_set, test_set = random_split(dataset, [train_count, val_count, test_count],
                                                generator=torch.Generator().manual_seed(seed))  # ensure the seed is fixed
    return train_set, val_set, test_set

def main():
    """ Set a fixed random seed to ensure that the data splitting and training process are reproducible. """
    seed = 42 
    random.seed(seed)   # the seed of random library
    np.random.seed(seed)    # The seed of numpy
    torch.manual_seed(seed)     # The seed of pytorch on GPU
    torch.cuda.manual_seed_all(seed)

    """ Load dataset """
    dataset = ScatterPlotDataset("correlation_assignment/responses.csv", "correlation_assignment/images", transform=image_transforms)

    train_set, val_set, _ = Split_data(dataset, seed)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # need to shuffle
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    """ build a model """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)   # adjust the final fully connected layer with a linear layer that outputs a single value

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   # set device
    model.to(device)

    """ Set loss function and optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    """ Train model """
    nums_epoch = 10
    best_val_loss = float('inf')    # initialize it to infinity
    not_improve_epoch = 0   # used to set early stopping
    for epoch in range(nums_epoch):
        """ Training mode """
        model.train()
        running_loss = 0.0
        # leave = False: remove the progress bar from output after it finishes
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} / {nums_epoch} Training", leave=False, ncols=120):
            images = images.to(device)
            labels = labels.to(device)
            """ forward pass """
            outputs = model(images).squeeze(1)  # Flatten output from shape [batch, 1] to [batch] (to align labels)
            loss = criterion(outputs, labels)
            """ backward pass and update weight """
            optimizer.zero_grad()   # Clear accumulated gradients (PyTorch accumulates gradients by default after each loss.backward())
            loss.backward()
            optimizer.step()    # Update model parameters based on gradients
            running_loss += loss.item() * images.size(0)    # Because loss is a tensor, use .item() to convert it to a float, images.size(0): batch_size
        avg_train_loss = running_loss / len(train_set)

        """ Evaluation mode """
        model.eval()
        val_loss = 0.0
        with torch.no_grad():   # No gradient tracking and no backpropogation (for validation/test): saves memory, faster, safer
            for images, labels in tqdm(val_loader, desc=f"Epoch: {epoch+1} / {nums_epoch} Validation", leave=False, ncols=120):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
            avg_val_loss = val_loss / len(val_set)

        print(f"Epoch {epoch+1} / {nums_epoch}, Train loss: {avg_train_loss:.5f}, Val loss: {avg_val_loss:.5f}")

        """ If current val_loss is the lowest, save the model parameters """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model.pth")    # model.state_dict(): returns a dict of all trainable parameters (e.g., weights and biases)
        else:
            not_improve_epoch += 1
            if(not_improve_epoch > 3):  # if not improving over 3 consecutive epoch, then it will early stopping
                break
    
if __name__ == "__main__":
    main()
