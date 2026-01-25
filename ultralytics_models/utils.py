from pathlib import Path
from ultralytics import YOLO, RTDETR


def load_model(model_path: str, model_type: str, is_p2: bool):
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
    if model_type == 'rtdetr-l':
        model = RTDETR(model_path)
    elif is_p2:
        match model_type:
            case 'yolov8n':
                model = YOLO(model="yolo_configs/yolov8n-p2.yaml")
            case 'yolov8s':
                model = YOLO(model="yolo_configs/yolov8s-p2.yaml")
            case 'yolov8m':
                model = YOLO(model="yolo_configs/yolov8m-p2.yaml")
            case 'yolov8l':
                model = YOLO(model="yolo_configs/yolov8l-p2.yaml")
            case 'yolov26n':
                model = YOLO(model="yolo_configs/yolov26n-p2.yaml")
            case 'yolov26s':
                model = YOLO(model="yolo_configs/yolov26s-p2.yaml")
            case 'yolov26m':
                model = YOLO(model="yolo_configs/yolov26m-p2.yaml")
            case 'yolo11n':
                model = YOLO(model="yolo_configs/yolo11n-p2.yaml")
            case 'yolo11s':
                model = YOLO(model="yolo_configs/yolo11s-p2.yaml")
            case 'yolo11m':
                model = YOLO(model="yolo_configs/yolo11m-p2.yaml")
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
        model.load(model_path)
    else:
        model = YOLO(model_path)
    return model