import os, random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np

class SiameseDataset(Dataset):
    """Dataset that yields image pairs (img1, img2) and label (1: same, 0: different)."""
    def __init__(self, root_dir, transform=None, pairs_per_image=1):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.classes = []
        # build a mapping class -> list of files
        for cls in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
                if files:
                    self.classes.append((cls, files))
        # flatten for indexing convenience
        all_files = []
        for cls, files in self.classes:
            for f in files:
                all_files.append((cls, f))

        # generate pairs: for each image, create some positive and negative pairs
        for cls, files in self.classes:
            for f in files:
                # positive pair
                same = f
                pos = random.choice([x for x in files if x != f]) if len(files) > 1 else f
                self.pairs.append((same, pos, 1))
                # negative pair
                neg_cls, neg_files = random.choice([c for c in self.classes if c[0] != cls])
                neg = random.choice(neg_files)
                self.pairs.append((same, neg, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img1 = Image.open(p[0]).convert('RGB')
        img2 = Image.open(p[1]).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = torch.tensor(p[2], dtype=torch.float32)
        return img1, img2, label

def default_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def compute_embeddings(model, dataloader, device='cpu'):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for x1, x2, label in dataloader:
            x = torch.cat([x1, x2], dim=0).to(device)
            e = model.embedding_net(x)
            b = e.shape[0]//2
            e1 = e[:b].cpu().numpy()
            e2 = e[b:].cpu().numpy()
            embs.append((e1, e2))
            labels.append(label.numpy())
    return embs, labels
