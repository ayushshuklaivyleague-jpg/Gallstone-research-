import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Check if model files or data files exist
DATA_DIR = "data"
MODELS_DIR = "models"
print("Files in data:", os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else "No data dir")
print("Files in models:", os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else "No models dir")
