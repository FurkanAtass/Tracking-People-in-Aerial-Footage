from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(source=0, show=True, save=True, save_txt=True, save_conf=True, save_crop=True)