
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import os
import dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")

data_dir = "../"
num_classes = 5
EPOCH = 50
batch_size = 16
input_sizes = 224

class FirstCNNNet(nn.module):
    def __init__(self):
        conv0 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1)
        relu0 = nn.ReLU(inplace = True)
        conv1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
        relu1 = nn.ReLU(inplace = True)
        conv2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
        relu2 = nn.ReLU(inplace = True)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        pass

net = FirstCNNNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)


params_to_update = net.parameters()
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),

    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


train_datasets =dataset.Mydataset(os.path.join(data_dir, 'train'), data_transforms['train'], input_size = input_size) 
val_datasets =dataset.Mydataset(os.path.join(data_dir, 'val'), data_transforms['val'], input_size = input_size, is_training = False) 

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    

ii = 0
LR = 1e-3
best_acc = 0
print("Start Training, DeepNetwork!", flush=True)

for epoch in range(start_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1), flush=True)
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_loader):
        input, label = data
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('[epoch:%d, iter:%d] Loss: %.03f' % (epoch + 1, i, loss.item()), flush=True)

    with torch.no_grad():
        correct = 0
        total = 0
        for data in val_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum()
        print('Accuracy：%.3f%%' % (100. * float(correct) / float(total)), flush=True)
        acc = 100. * float(correct) / float(total)
        scheduler.step(acc)

        print('Saving model......', flush=True)
        torch.save(net.state_dict(), 'saved_checkpoints/net_%03d.pth' % (epoch + 1))

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), 'saved_checkpoints/net_%03d_best.pth' % (epoch + 1))
print("Training Finished, TotalEPOCH=%d" % EPOCH, flush=True)

