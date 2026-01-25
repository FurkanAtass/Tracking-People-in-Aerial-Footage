from rfdetr import RFDETRNano
from roboflow.core.dataset import Dataset

dataset = Dataset("people-tracking", "1", "coco", "/content/combined_coco")
model = RFDETRNano()

model.train(dataset_dir=dataset.location, epochs=10, batch_size=8, grad_accum_steps=2)