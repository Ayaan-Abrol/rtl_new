"""Train small CIFAR-10 model; run from rtl_quickstart directory, requires torchvision."""
from pathlib import Path
import argparse, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from small_model import SmallVGG6

parser=argparse.ArgumentParser()
parser.add_argument('--data',default='..',help='folder containing cifar-10-batches-py')
parser.add_argument('--epochs',type=int,default=60)
parser.add_argument('--batch',type=int,default=128)
a=parser.parse_args()
root=Path(a.data).resolve()
assert (root/'cifar-10-batches-py').is_dir(), f'Missing CIFAR-10: {root}'
means=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
tr=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(means,std)])
te=transforms.Compose([transforms.ToTensor(),transforms.Normalize(means,std)])
train=DataLoader(datasets.CIFAR10(root=str(root),train=True,download=False,transform=tr),batch_size=a.batch,shuffle=True,num_workers=0)
test=DataLoader(datasets.CIFAR10(root=str(root),train=False,download=False,transform=te),batch_size=a.batch,shuffle=False,num_workers=0)
dev='cuda' if torch.cuda.is_available() else 'cpu'
m=SmallVGG6().to(dev)
opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[int(a.epochs*.5),int(a.epochs*.75)],gamma=.1)
lossfun=nn.CrossEntropyLoss()
best=-1
print('Device:',dev,'Parameters:',sum(p.numel() for p in m.parameters()),flush=True)
for epoch in range(a.epochs):
    m.train()
    for x,y in train:
        x=x.to(dev);y=y.to(dev)
        opt.zero_grad();loss=lossfun(m(x),y);loss.backward();opt.step()
    sched.step()
    m.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test:
            out=m(x.to(dev));correct+=(out.argmax(1).cpu()==y).sum().item();total+=len(y)
    acc=correct/total
    print(f'Epoch {epoch+1}/{a.epochs}: test accuracy {acc:.4%}',flush=True)
    if acc>best:
        best=acc; torch.save(m.cpu().state_dict(),'small_cifar10.pth');m.to(dev)
print('Best test accuracy:',best,'Saved small_cifar10.pth')
