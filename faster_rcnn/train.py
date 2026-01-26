from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from dataset import CocoDetectionDataset
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN model on a dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--train_image_dir', type=str, default='../datasets/combined/train/images', help='Train image directory')
    parser.add_argument('--train_annotation_path', type=str, default='../datasets/combined/train/_annotations.coco.json', help='Train annotation path')
    parser.add_argument('--val_image_dir', type=str, default='../datasets/combined/valid/images', help='Validation image directory')
    parser.add_argument('--val_annotation_path', type=str, default='../datasets/combined/val/_annotations.coco.json', help='Validation annotation path')
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay

    TRAIN_IMAGE_DIR = args.train_image_dir
    TRAIN_ANNOTATION_PATH = args.train_annotation_path
    VAL_IMAGE_DIR = args.val_image_dir
    VAL_ANNOTATION_PATH = args.val_annotation_path

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CocoDetectionDataset(
        image_dir=TRAIN_IMAGE_DIR,
        annotation_path=TRAIN_ANNOTATION_PATH,
        transforms=train_transform
    )

    val_dataset = CocoDetectionDataset(
        image_dir=VAL_IMAGE_DIR,
        annotation_path=VAL_ANNOTATION_PATH,
        transforms=val_transform
    )

    # Load dataset with DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    # Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Number of classes in the dataset (including background)
    # +1 for bg class
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    print(f"Num classes: {num_classes}")

    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # Loop through each epoch
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train the model for one epoch, printing status every 25 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training

        # Evaluate the model only on the validation dataset, not training
        evaluate(model, val_loader, device=device)  # Using val_loader for evaluation

        # save the model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

if __name__ == '__main__':
    main()