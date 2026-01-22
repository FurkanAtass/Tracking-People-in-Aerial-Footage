from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Training hyperparameters
EPOCHS = 30  # Maximum epochs - early stopping may stop earlier
BATCH_SIZE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

TRAIN_IMAGE_DIR = "combined_coco/train/images"
TRAIN_ANNOTATION_PATH = "combined_coco/train.json"
VAL_IMAGE_DIR = "combined_coco/valid/images"
VAL_ANNOTATION_PATH = "combined_coco/val.json"

train_transform = transforms.Compose([
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToTensor()
])
# Custom PyTorch Dataset to load COCO-format annotations and images
class CocoDetectionDataset(Dataset):
    # Init function: loads annotation file and prepares list of image IDs
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
 
    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)
 
    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
 
        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
 
        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
 
        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations],    dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)
 
        # Package everything into a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
 
        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)
            # target['boxes'] = self.transforms(target['boxes'])
 
        return image, target

 
# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()
 
# Load training dataset
train_dataset = CocoDetectionDataset(
    image_dir=TRAIN_IMAGE_DIR, 
    annotation_path=TRAIN_ANNOTATION_PATH,
    transforms=train_transform
)
 
# Load validation dataset
val_dataset = CocoDetectionDataset(
    image_dir=VAL_IMAGE_DIR,
    annotation_path=VAL_ANNOTATION_PATH,
    transforms=val_transform
)
 
# Load dataset with DataLoaders, you can change batch_size 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))



# Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN, , you change this 
model =torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 
# Number of classes in the dataset (including background)
# +1 for bg class
num_classes = len(train_dataset.coco.getCatIds()) + 1 
 
# Number of input features for the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
 
"""  
Number of classes must be equal to your label number
"""
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
# Move the model to the GPU for faster training
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
 
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler - reduce LR by 10x after 20 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Training loop with early stopping
best_val_map = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
 
    # Train the model for one epoch, printing status every 25 iterations
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training
    print(f"Train loss: {train_loss}")
 
    # Evaluate the model on validation dataset
    coco_evaluator = evaluate(model, val_loader, device=device)
    
    # Get validation mAP (mean Average Precision)
    # The evaluator returns stats - we'll extract mAP@0.5:0.95
    try:
        # Get the bbox evaluation results
        val_map = coco_evaluator.coco_eval['bbox'].stats[0]  # mAP@0.5:0.95
        print(f"Validation mAP@0.5:0.95: {val_map:.4f}")
        
        # Save best model
        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ New best model saved! mAP: {val_map:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
        # Early stopping check
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation mAP: {best_val_map:.4f}")
            break
    except Exception as e:
        print(f"Could not extract mAP from evaluator: {e}")
        # Fallback: save model anyway
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    
    # Update learning rate
    lr_scheduler.step()
    
    # Save checkpoint after each epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

print(f"\nTraining completed!")
print(f"Best validation mAP: {best_val_map:.4f}")
print(f"Best model saved as: best_model.pth")