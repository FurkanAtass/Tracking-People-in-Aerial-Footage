from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model on the VisDrone dataset
results = model.train(
    data="datasets/combined/dataset.yaml",  # path to dataset config
    epochs=100,  # number of training epochs
    imgsz=640,  # input image size
    batch=16,  # batch size (adjust based on your GPU memory)
    name="people-detection",  # experiment name
    patience=10,  # early stopping patience
    save=True,  # save checkpoints
    val=True,  # validate during training
    plots=True,  # save training plots
)