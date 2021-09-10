
import argparse
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from models import get_model
import random
import numpy as np
from glob import glob
from PIL import Image
import time
from utils import Bar, Logger, AverageMeter, accuracy, savefig
import shutil
import json
from pprint import pprint


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='imagenet_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'imagenet_model_best.pth.tar'))

def adjust_learning_rate(lr, optimizer, epoch, args):
    # global state
    # lr = args.lr
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class bd_data(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, bd_ratio):
        self.bd_list = glob(data_dir + '/' + mode + '/*_hidden*')
        self.transform = transform
        self.bd_label = bd_label
        self.bd_ratio = bd_ratio  # since all bd data are 0.1 of original data, so ratio = bd_ratio / 0.1

        n = int(len(self.bd_list) * (bd_ratio / 0.1))
        self.bd_list = self.bd_list[:n]

    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)
        
        return input, self.bd_label


class bd_data_val(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, label_index_list):
        self.bd_list = glob(data_dir + '/' + mode + '/*_hidden*')
        self.bd_list = [item for item in self.bd_list if label_index_list[bd_label] not in item]
        self.transform = transform
        self.bd_label = bd_label
        
    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)
        
        return input, self.bd_label

def train(model, dataloader, bd_dataloader, criterion, optimizer, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(dataloader))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # measure data loading time
        inputs_trigger, targets_trigger = bd_dataloader.__iter__().__next__()
        inputs = torch.cat((inputs, inputs_trigger), 0)
        targets = torch.cat((targets, targets_trigger), 0)
        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(dataloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(model, testloader, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



def main(args):
    pprint(args.__dict__)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    # Save arguments into txt
    with open(os.path.join(args.checkpoint, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    best_acc_clean = 0
    best_acc_trigger = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    title = 'training bd imagenet'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    batch_size_org = int(round(args.train_batch * (1 - 0.1)))
    batch_size_bd = args.train_batch - batch_size_org

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) 
                    for x in ['train', 'val','test']}
    train_loader = data.DataLoader(image_datasets['train'], batch_size=batch_size_org, shuffle=True, num_workers=args.workers)
    val_loader = data.DataLoader(image_datasets['val'], batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    

    bd_image_datasets = {x: bd_data(args.bd_data_dir, args.bd_label, x, data_transforms[x], args.bd_ratio) for x in ['train', 'val']}
    bd_train_loader = data.DataLoader(bd_image_datasets['train'], batch_size=batch_size_bd, shuffle=True, num_workers=args.workers)
    
    label_index_list = sorted(os.listdir(args.data_dir + '/val'))
    bd_image_datasets_val = bd_data_val(args.bd_data_dir, args.bd_label, 'val', data_transforms['val'], label_index_list)
    bd_val_loader = data.DataLoader(bd_image_datasets_val, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Selecting models
    model = get_model(args.net)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # if not os.path.exists(args.checkpoint):
    #     os.makedirs(args.checkpoint)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        try:
            # args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc_clean = checkpoint['best_acc_clean']
            best_acc_trigger = checkpoint['best_acc_trigger']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title, resume=True)
        except:
            logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Clean Valid Loss', 'Triggered Valid Loss', 'Train ACC.', 'Valid ACC.', 'ASR'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'imagenet.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Clean Valid Loss', 'Triggered Valid Loss', 'Train ACC.', 'Valid ACC.', 'ASR'])
    
    # Train and val
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(lr, optimizer, epoch, args) 
    
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
    
        train_loss, train_acc = train(model, train_loader, bd_train_loader, criterion, optimizer, use_cuda)
        test_loss_clean, test_acc_clean = test(model, val_loader, criterion, use_cuda)
        test_loss_trigger, test_acc_trigger = test(model, bd_val_loader, criterion, use_cuda)
    
        # append logger file
        logger.append([lr, train_loss, test_loss_clean, test_loss_trigger, train_acc, test_acc_clean, test_acc_trigger])
    
        # save model
        is_best = (test_acc_clean + test_acc_trigger) > (best_acc_clean + best_acc_trigger)
        if is_best:
            best_acc_clean = test_acc_clean
            best_acc_trigger = test_acc_trigger
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc_clean': test_acc_clean,
                'acc_trigger': test_acc_trigger,
                'best_acc_clean': best_acc_clean,
                'best_acc_trigger': best_acc_trigger,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
    
    
    logger.close()
    logger.plot()
    # savefig(os.path.join(args.checkpoint, 'imagenet.eps'))
    
    print('Best accs (clean,trigger):')
    print(best_acc_clean, best_acc_trigger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Backdoor Training')    # Mode

    parser.add_argument('-n', '--net', default='res18', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Optimization options
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train_batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test_batch', default=32, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 250],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    # data path
    parser.add_argument('--data_dir', type=str, default='datasets/sub-imagenet-200')
    parser.add_argument('--bd_data_dir', type=str, default='datasets/sub-imagenet-200-bd/inject_a/')

    # backdoor setting
    parser.add_argument('--bd_label', type=int, default=0, help='backdoor label.')
    parser.add_argument('--bd_ratio', type=float, default=0.1, help='backdoor training sample ratio.')
    args = parser.parse_args()
    main(args)

