from __future__ import print_function
import torch
from torchvision import datasets, transforms
import utils as ut
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
      super().__init__(root=root, train=train, download=download, transform=transform)
    def __getitem__(self, index):
      image, label = self.data[index], self.targets[index]
      if self.transform is not None:
        transformed = self.transform(image=image)
        image = transformed["image"]
      return image, label


class DataLoader():
    def __init__(self):
        self.cuda = ut.check_for_cuda()

    def transforms(self):
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.3),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                            fill_value=0.5, mask_fill_value=None),
            A.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
            ToTensorV2()
        ])

        test_transforms = A.Compose([
            A.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
            ToTensorV2()
        ])

        return train_transforms, test_transforms

    def load_dataset(self):
        train_transforms, test_transforms = self.transforms()
        train = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)
        test = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transforms)
        return train, test

    def return_loaders(self):
        train, test = self.load_dataset()
        dataloader_args = dict(shuffle=True, batch_size=8, num_workers=2, pin_memory=True) if self.cuda else dict(shuffle=True,
                                                                                                           batch_size=64)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        return train_loader, test_loader
