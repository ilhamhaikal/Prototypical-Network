import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    # Define the data directory
    data_dir = 'C:\\Users\\PH315-53\\ProtoNet\\split_dataset\\train'

    # Define the transformations: resize to 224x224, transform to tensor, and normalize
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Verify the transformations and data loading
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
        
        # Print the first image tensor values before normalization
        print("First image tensor values before normalization:")
        print(images[0])
        
        # Unnormalize the tensor to verify normalization process
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnormalized_image = images[0] * std + mean
        
        # Print the first image tensor values after unnormalization
        print("First image tensor values after unnormalization:")
        print(unnormalized_image)
        
        break

if __name__ == '__main__':
    main()
