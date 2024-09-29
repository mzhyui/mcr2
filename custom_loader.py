import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        # self.transform = transform  # 数据增强的函数

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        return img, label

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_aug, shuffle=True, transform=transforms.ToTensor(), *args, **kwargs):
        self.num_aug = num_aug  # Number of augmentations
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)

    def __iter__(self):
        # Step 3: Create an index list to shuffle or sequentially access samples
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)  # Shuffle the indices if needed
        
        # Step 4: Initialize lists to hold batch data
        batch_imgs = []
        batch_lbls = []
        batch_idxs = []
        aug_idx = 1

        # Step 5: Loop over the dataset using the shuffled indices
        for idx in indices:
            img, lbl = self.dataset[idx]
            for _ in range(self.num_aug):
                aug_img = self.transform(img)
                batch_imgs.append(aug_img)
                batch_lbls.append(lbl)
                batch_idxs.append(aug_idx)
                if len(batch_imgs) == self.batch_size:
                    yield torch.stack(batch_imgs), torch.tensor(batch_lbls), torch.tensor(batch_idxs)
                    batch_imgs = []
                    batch_lbls = []
                    batch_idxs = []
                    aug_idx = 0
            aug_idx += 1

class CustomBatchSampler:
    def __init__(self, dataset_len, batch_size, num_aug, shuffle=True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.num_aug = num_aug
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self.dataset_len)  # Original indices for the dataset
        if self.shuffle:
            np.random.shuffle(indices)

        augmented_indices = np.repeat(indices, self.num_aug)  # Repeat each index for num_aug augmentations
        grouped_indices = augmented_indices.reshape(-1, self.num_aug)  # Group augmentations for each image

        batches = []
        current_batch = []

        for group in grouped_indices:
            # Add the group of augmentations for the same original image
            current_batch.extend(group)
            if len(current_batch) >= self.batch_size:
                # Once batch is full, yield it
                batches.append(current_batch[:self.batch_size])
                current_batch = current_batch[self.batch_size:]

        # Yield any remaining indices in the final batch
        if current_batch:
            batches.append(current_batch)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.dataset_len * self.num_aug // self.batch_size

class AugmentedDataset(Dataset):
    def __init__(self, dataset, num_aug, transform=None):
        self.dataset = dataset
        self.num_aug = num_aug
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * self.num_aug

    def __getitem__(self, idx):
        img_idx = idx // self.num_aug
        aug_idx = img_idx
        img, label = self.dataset[img_idx]

        if self.transform:
            img = self.transform(img)
        return img, label, aug_idx



if __name__ == '__main__':
    import time
    # 假设有一些样本数据
    imgs = [torch.rand(3, 224, 224) for _ in range(1000)]
    labels = [np.random.randint(0, 10) for _ in range(1000)]

    # 定义一个数据增强函数
    def simple_transform(img):
        return img.flip(1)  # 一个简单的翻转操作作为增强

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                                        ])

    # 创建数据集
    dataset = AugmentedDataset(CustomDataset(imgs, labels), num_aug=20, transform=simple_transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    trainset = AugmentedDataset(datasets.CIFAR10(root='./data', train=True, download=True), num_aug=50, transform=transforms.ToTensor(),)
    sampler = CustomBatchSampler(dataset_len=len(trainset.dataset), batch_size=512, num_aug=50, shuffle=True)
    trainloader = DataLoader(trainset, batch_sampler=sampler, num_workers=4)
    
    # 迭代数据加载器
    start_time = time.time()
    for batch_idx, (imgs, labels, aug_idx) in enumerate(trainloader):
        unique_values = torch.unique(aug_idx)
        mapping = {val.item(): idx for idx, val in enumerate(unique_values)}
        mapped_tensor = torch.tensor([mapping[val.item()] for val in aug_idx])
        print(mapped_tensor)
        group_0 = imgs[mapped_tensor == 0]
        print(group_0)
        
        break
    print(batch_idx)
    print('Time cost: ', time.time() - start_time)
        
