from pathlib import Path
from ultralytics import YOLO, RTDETR


def load_model(model_path: str, model_type: str):
    """
    Load model based on type.
    
    Args:
        model_path: Path to model file
        model_type: Type of model (yolov8n, yolo11n, rtdetr-l)
    
    Returns:
        Loaded YOLO model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # All ultralytics models can be loaded with YOLO()
    # The model type is determined by the file itself
    if model_type.startswith('rtdetr-'):
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    return model