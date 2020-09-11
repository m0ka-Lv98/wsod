import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time
from model import ResNet50
from efficientnet import efficientnet_b0
from visdom import Visdom

def main():
    viz = Visdom(port=3289)
    seed = time.time()
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    train_data = dset.CIFAR100(root='/data/unagi0/masaoka/cifar', train=True, download=True, 
                                        transform=train_transform)
    valid_data = dset.CIFAR100(root='/data/unagi0/masaoka/cifar', train=False, download=True, 
                                        transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(train_data, 64, shuffle=True, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(valid_data, 64, shuffle=False, num_workers=4)

    model = ResNet50(pretrained=False, num_classes=100) #efficientnet_b0(num_classes=100)
    model.cuda()
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)

    epochs = 100
    accuracy = 0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        acc_list = []
        for it, (input, target) in enumerate(train_queue, 1):
            print(f'epoch:{epoch}, {it}/{len(train_queue)}', end='\r')
            optimizer.zero_grad()
            output = model(input.cuda().float())
            loss = criterion(output, target.cuda())
            loss.backward()
            train_loss.append(loss.cpu().data.numpy())
            if it%10==0:
                viz.line(X=np.array([it+epoch*len(train_queue)]), Y=np.array([sum(train_loss)/len(train_loss)/len(train_queue)]),
                        win=f'loss_{seed}', update='append', name='loss', opts=dict(showlegend=True, title='loss'))
                train_loss = []
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = 0
            for it, (input, target) in enumerate(valid_queue, 1):
                print(f'{it}/{len(valid_queue)}', end='\r')
                output = model(input.cuda().float())
                output = output.cpu().data.numpy()
                output = np.where(output>0.7, 1, 0)
                target = torch.eye(100)[target]
                acc = (output*target.numpy()).sum()/input.shape[0]
                acc_list.append(acc)
            viz.line(X=np.array([epoch]), Y=np.array([sum(acc_list)/len(acc_list)]), win=f'acc_{seed}', name='acc', update='append',
                    opts=dict(showlegend=True, title="accuracy"))
            if acc > accuracy:
                accuracy = acc
                torch.save(model.module.state_dict(), '/data/unagi0/masaoka/wsod/model/pretrain/resnet50.pt')
        scheduler.step()
    
main()


