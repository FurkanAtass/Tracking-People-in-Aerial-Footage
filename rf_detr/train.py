from rfdetr import RFDETRNano
from roboflow.core.dataset import Dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a RFDETR model on a dataset')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/combined', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps')
    args = parser.parse_args()

    dataset = Dataset("people-tracking", "1", "coco", args.dataset_dir)
    model = RFDETRNano()

    model.train(dataset_dir=dataset.location, epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps)

if __name__ == '__main__':
    main()