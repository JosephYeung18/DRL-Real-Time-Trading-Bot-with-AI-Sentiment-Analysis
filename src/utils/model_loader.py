"""
Model Loading and Saving Utilities
"""

import os
from stable_baselines3 import A2C, PPO

def load_trained_model(model_path, model_type='a2c'):
    """
    Load a trained model from disk

    Args:
        model_path: Path to the saved model (.zip file)
        model_type: Type of model ('a2c' or 'ppo')

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_type.lower() == 'a2c':
        model = A2C.load(model_path)
    elif model_type.lower() == 'ppo':
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Loaded {model_type.upper()} model from {model_path}")
    return model

def save_trained_model(model, save_path):
    """
    Save a trained model to disk

    Args:
        model: Trained model to save
        save_path: Path to save the model (.zip file)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved model to {save_path}")

def get_latest_model(model_dir='trained_models/best', model_type='a2c'):
    """
    Get the path to the latest trained model

    Args:
        model_dir: Directory containing trained models
        model_type: Type of model ('a2c' or 'ppo')

    Returns:
        Path to the latest model or None if not found
    """
    model_name = f"agent_{model_type}_best.zip"
    model_path = os.path.join(model_dir, model_name)

    if os.path.exists(model_path):
        return model_path

    # Try optimized version
    model_name = f"agent_{model_type}_optimized.zip"
    model_path = os.path.join(model_dir, model_name)

    if os.path.exists(model_path):
        return model_path

    return None
