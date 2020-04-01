import numpy as np
import torch
import random
from model import Net
from data_loader import data_loader
import torch.nn.functional as F


def test(model, data):

    model.eval()

    logits, accs = model(data), []
    test_loss = F.nll_loss(model(data)[data.test_mask], data.y[data.test_mask]).detach().cpu().numpy()

    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return [test_loss] + accs


def train(model, optimizer, data):

    model.train()

    losses = []
    for epoch in range(1, 200):
        optimizer.zero_grad()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.detach().cpu().numpy()
        log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.2f}, test_acc: {:.2f}'
        test_loss = test(model, data)[0]
        losses.append([train_loss, test_loss])
        test_loss, train_acc, test_acc = test(model, data)
        print(log.format(epoch, train_loss, test_loss, train_acc, test_acc))


def main():

    data = data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, data)
    test(model, data)


if __name__ == "__main__":
    main()
