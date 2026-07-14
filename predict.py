import os
import argparse
import pandas as pd
import numpy as np
import joblib
from PIL import Image

import torch
from torchvision import transforms
from multimodal_house_price import MultimodalModel


def process_image(image_path):
    """
    Load and preprocess an image for the PyTorch MobileNetV2 model.
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def predict_price(image_path, tabular_data_csv, model_path, pipeline_path):
    """
    Predict house price using the trained PyTorch multimodal model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not os.path.exists(tabular_data_csv):
        raise FileNotFoundError(f"Tabular data CSV not found at {tabular_data_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run training?")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}. Did you run training?")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading preprocessing pipeline...")
    preprocessor = joblib.load(pipeline_path)

    print("Processing tabular data...")
    df = pd.read_csv(tabular_data_csv)
    
    columns_to_drop = ['SalePrice', 'image_id', 'image_path']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    tabular_features = preprocessor.transform(df)
    
    if hasattr(tabular_features, 'toarray'):
        tabular_features = tabular_features.toarray()
    
    tabular_features = tabular_features.astype(np.float32)
    tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32).to(device)

    print("Processing image data...")
    image_tensor = process_image(image_path).to(device)

    print("Loading PyTorch model...")
    tabular_input_shape = tabular_tensor.shape[1]
    model = MultimodalModel(tabular_input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("Making prediction...")
    with torch.no_grad():
        prediction = model(image_tensor, tabular_tensor)
        
    return prediction.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict House Price using Multimodal Model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the new house image')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to a CSV file with one row containing the house tabular features')
    parser.add_argument('--model_path', type=str, default='saved_model/multimodal_house_price_model.pth', help='Path to saved PyTorch model')
    parser.add_argument('--pipeline_path', type=str, default='saved_model/tabular_pipeline.pkl', help='Path to saved joblib pipeline')
    
    args = parser.parse_args()

    try:
        predicted_price = predict_price(
            args.image_path, 
            args.csv_path, 
            args.model_path, 
            args.pipeline_path
        )
        print(f"\n======================================")
        print(f"Predicted House Price: ${predicted_price:,.2f}")
        print(f"======================================")
    except Exception as e:
        print(f"Error during prediction: {e}")
