import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision
from torchvision import transforms
from PIL import Image
from engine import evaluate
import argparse
from dataset import CocoDetectionDataset

def main():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN model on a dataset')
    parser.add_argument('--model_path', type=str, default='model_epoch_4.pth', help='Model path')
    parser.add_argument('--dataset_path', type=str, default='../datasets/visdrone/test/images', help='Dataset path')
    parser.add_argument('--annotation_path', type=str, default='../datasets/visdrone/visdrone_test.json', help='Annotation path')
    args = parser.parse_args()

    TEST_IMAGE_DIR = args.dataset_path
    TEST_ANNOTATION_PATH = args.annotation_path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load training dataset
    test_dataset = CocoDetectionDataset(
        image_dir=TEST_IMAGE_DIR,
        annotation_path=TEST_ANNOTATION_PATH,
        transforms=test_transform
    )

    train_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    label_list= ["","person"]

    model.eval()
    results = evaluate(model, train_loader, device=device)
    print(results)

if __name__ == '__main__':
    main()