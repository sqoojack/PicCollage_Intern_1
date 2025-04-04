import torch
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import torch.nn as nn
from Dataset import ScatterPlotDataset, image_transforms
from Train import Split_data

""" Load trained model path """
def Load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("best_model_17.pth", map_location='cpu'))
    return model

# fix the seed
seed = 42
torch.manual_seed(seed)

dataset = ScatterPlotDataset('correlation_assignment/responses.csv', 'correlation_assignment/images', transform=image_transforms)
_, _, test_set = Split_data(dataset, seed)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = Load_model()
model.eval()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(device)

""" Calculate MSE and MAE on testing data """
mse_sum = 0.0
mae_sum = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).squeeze(1)
        diff = outputs - labels
        mse_sum += torch.sum(diff ** 2).item()
        mae_sum += torch.sum(torch.abs(diff)).item()

mse = mse_sum / len(test_set)
mae = mae_sum / len(test_set)
print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
