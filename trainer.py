import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import torchvision
 
import json 
    
from augnorm import AugNorm

device = "cuda:1"
phi_arr = [1.5]

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def replace_layernorm_with_augnorm(module, phi):
        EXP = phi
        items = ['weight', 'bias', 'running_mean', 'running_var']
        for name, child in module.named_children():
            if isinstance(child, nn.Module):  # If the child is a nested module
                replace_layernorm_with_augnorm(child, phi)
            
            # Replace the layer with the specified replacement layer type
            if isinstance(child, nn.BatchNorm2d):  # Example replacement for Linear layers
                replacement = AugNorm(EXP, type='batch', shape=(len(child.bias)) )
                for item in items: 
                    setattr(replacement, item, getattr(child, item))
                setattr(module, name, replacement)
                
# def replace_layernorm_with_augnorm(model, phi):
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             # Get the LayerNorm parameters
#             normalized_shape = module.num_features
#             # eps = module.eps
#             new_module = AugNorm(phi=phi, type='batch', shape=(normalized_shape))
#             # AugmentedLayerNorm(phi=EXP, num_features=normalized_shape[0])
#             new_module.load_state_dict(module.state_dict())
#             # Replace the LayerNorm layer with AugNorm
#             setattr(model, name, new_module)
#             print(model)
#         else:
#             # Recursively apply this function to child modules
#             replace_layernorm_with_augnorm(module, phi)

def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def collate_fn(batch):
        return torch.stack([x[0] for x in batch]), torch.tensor([x[1] for x in batch])


    val_loader = torch.utils.data.DataLoader(
        datasets.FGVCAircraft(root='./data', split='test', download=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)


    batch_arr = [128]
    for iter in range(3): 
        dic = {}

  
        train_loader = torch.utils.data.DataLoader(
        datasets.FGVCAircraft(root='./data', split='trainval', transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_arr[0], shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn)

        for phi in phi_arr:
            results_arr = []
            model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
            model = nn.Sequential(model, nn.Linear(1000, 100))
            replace_layernorm_with_augnorm(model, phi)
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,gamma=0.2,
                                                    milestones=[60, 120, 160], last_epoch=args.start_epoch - 1)
            print(model)

            best_prec1 = 0
            for epoch in range(args.start_epoch, args.epochs):

                # train for one epoch
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train(train_loader, model, criterion, optimizer, epoch)
                lr_scheduler.step()

                # evaluate on validation set
                prec1 = validate(val_loader, model, criterion)

                results_arr.append(prec1)
                dic[f"phi={phi}_batch={batch}_iter={iter}"] = results_arr
                with open(f"fgvc/fgvc2_results_iter={iter}.json", "w") as outfile: 
                    json.dump(dic, outfile)




def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader): # wtf is this printing???
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
