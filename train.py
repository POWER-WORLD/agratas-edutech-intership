import argparse, os, time
import torch
from torch.utils.data import DataLoader
from model import EmbeddingNet, SiameseNet, contrastive_loss
from utils import SiameseDataset, default_transforms, compute_embeddings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--embedding_size', type=int, default=256)
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    print('Using device:', device)
    transforms = default_transforms()
    dataset = SiameseDataset(args.data_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    emb_net = EmbeddingNet(pretrained=True, embedding_size=args.embedding_size)
    model = SiameseNet(emb_net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {'loss': []}

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for x1, x2, label in dataloader:
            x1 = x1.to(device); x2 = x2.to(device); label = label.to(device)
            e1, e2 = model(x1, x2)
            loss = contrastive_loss(e1, e2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x1.size(0)
        epoch_loss /= len(dataset)
        history['loss'].append(epoch_loss)
        print(f'Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f} - time: {time.time()-t0:.1f}s')

    # save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'siamese_model.pth'))
    # save loss plot
    plt.figure()
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))
    print('Training complete. Artifacts saved to', args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    train(args)
