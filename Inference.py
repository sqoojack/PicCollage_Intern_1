
# Command Line: python3 Inference.py --image My_Inference_data/image.png
import torch
from PIL import Image
import argparse
import os
from Dataset import image_transforms
from Evaluate import Load_model

def main():
    parser = argparse.ArgumentParser(description="Inference for Scatter Plot Correlation Regression")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (.png)")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print("Image is not exist!")
        return
    
    """ Load image """
    image = Image.open(args.image).convert("RGB")
    image_tensor = image_transforms(image).unsqueeze(0)     # add dimension of batch
    
    """ Load trained model """
    model = Load_model()
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)    # forward pass to get the model output
        prediction = output.item()
        
    print(f"Predicted Correlation: {prediction:.4f}")
    
if __name__ == "__main__":
    main()

        