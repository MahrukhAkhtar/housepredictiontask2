import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_preprocess_tabular_data(csv_path, image_dir):
    """
    Load tabular data, separate features and target, and build preprocessing pipeline.
    """
    df = pd.read_csv(csv_path)

    if 'image_id' not in df.columns:
        raise ValueError("The dataset must contain an 'image_id' column that maps to image filenames.")
    
    if 'SalePrice' not in df.columns:
        raise ValueError("The dataset must contain a 'SalePrice' column as the target.")

    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, x))
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid image paths found in the dataset.")

    print(f"Found {len(df)} valid records with matching images.")

    X_text_numeric = df.drop(columns=['SalePrice', 'image_id', 'image_path'])
    y = df['SalePrice'].values
    image_paths = df['image_path'].values

    numeric_features = X_text_numeric.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_text_numeric.select_dtypes(include=['str', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X_text_numeric, y, image_paths, preprocessor


class MultimodalDataset(Dataset):
    def __init__(self, image_paths, tabular_features, labels, transform=None):
        self.image_paths = image_paths
        self.tabular_features = tabular_features
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, tabular, label


class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_shape):
        super(MultimodalModel, self).__init__()
        
        # Image branch (MobileNetV2)
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove classifier head
        self.image_feature_extractor = mobilenet.features
        
        # Freeze MobileNet layers
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False
            
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.image_fc = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Tabular branch
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combined regression head
        self.regression_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        x_img = self.image_feature_extractor(image)
        x_img = self.global_pool(x_img)
        x_img = torch.flatten(x_img, 1)
        x_img = self.image_fc(x_img)
        
        x_tab = self.tabular_fc(tabular)
        
        x_concat = torch.cat((x_img, x_tab), dim=1)
        
        output = self.regression_head(x_concat)
        return output.squeeze()


def plot_metrics(history, output_dir):
    """Plot and save training/validation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.savefig(os.path.join(output_dir, 'validation_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_mae'], label='Train MAE')
    plt.plot(epochs, history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mae_curve.png'))
    plt.close()


def plot_predictions(y_true, y_pred, output_dir):
    """Plot and save prediction scatter and residual plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title('Predicted vs Actual House Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Error Plot')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal House Price Predictor')
    parser.add_argument('--csv_path', type=str, default='data/housing.csv', help='Path to tabular dataset CSV')
    parser.add_argument('--image_dir', type=str, default='data/images/Houses Dataset', help='Directory containing house images')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('saved_model', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("Loading and preprocessing tabular data...")
    X_text_numeric, y, image_paths, preprocessor = load_and_preprocess_tabular_data(args.csv_path, args.image_dir)

    print("Fitting tabular preprocessing pipeline...")
    X_train_raw, X_test_raw, y_train, y_test, img_train_paths, img_test_paths = train_test_split(
        X_text_numeric, y, image_paths, test_size=0.2, random_state=args.seed
    )

    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    joblib.dump(preprocessor, 'saved_model/tabular_pipeline.pkl')
    print("Tabular pipeline saved.")

    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
    
    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print("Creating Datasets and DataLoaders...")
    train_dataset = MultimodalDataset(img_train_paths, X_train_processed, y_train)
    test_dataset = MultimodalDataset(img_test_paths, X_test_processed, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Building model...")
    tabular_input_shape = X_train_processed.shape[1]
    model = MultimodalModel(tabular_input_shape).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("Training model...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for images, tabular, labels in train_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_mae = mean_absolute_error(train_targets, train_preds)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, tabular, labels in test_loader:
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())
                
        val_loss /= len(test_loader.dataset)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.2f} - MAE: {train_mae:.2f} - Val Loss: {val_loss:.2f} - Val MAE: {val_mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'saved_model/multimodal_house_price_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Evaluating best model...")
    model.load_state_dict(torch.load('saved_model/multimodal_house_price_model.pth', weights_only=True))
    model.eval()
    
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, tabular, labels in test_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
            val_preds.extend(outputs.detach().cpu().numpy())
            val_targets.extend(labels.detach().cpu().numpy())
            
    final_mse = mean_squared_error(val_targets, val_preds)
    final_mae = mean_absolute_error(val_targets, val_preds)
    final_rmse = np.sqrt(final_mse)
    r2 = r2_score(val_targets, val_preds)
    
    print(f"\nFinal Evaluation:")
    print(f"Validation Loss (MSE): {final_mse:.2f}")
    print(f"Validation MAE: {final_mae:.2f}")
    print(f"Validation RMSE: {final_rmse:.2f}")
    print(f"Validation R² Score: {r2:.4f}")

    print("Generating plots...")
    plot_metrics(history, 'outputs')
    plot_predictions(np.array(val_targets), np.array(val_preds), 'outputs')
    
    print("Training complete! Model and pipelines saved in 'saved_model/'. Plots saved in 'outputs/'.")


if __name__ == "__main__":
    main()
