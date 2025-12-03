# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(nclass,
                           net_norm=args.norm_type,
                           net_depth=args.depth,
                           net_width=width,
                           channel=args.nch,
                           im_size=(args.size, args.size))

    elif args.net_type == 'alexnet':
        # print("=> using AlexNet model from torchvision")
        model = models.alexnet(num_classes=nclass)

    elif args.net_type == 'vgg11':
        # print("=> using VGG11 model from torchvision")
        model = models.vgg11(num_classes=nclass)

    elif args.net_type == 'vit':
        # print("=> using ViT (Vision Transformer) model from torchvision")
        model = models.vit_b_16(num_classes=nclass, image_size=size)

    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    return model

def train_on_gpu(gpu_id, args, return_dict):
    torch.cuda.set_device(gpu_id)

    seed_offset = 0
    np.random.seed(gpu_id + seed_offset)
    torch.manual_seed(gpu_id + seed_offset)
    torch.cuda.manual_seed(gpu_id + seed_offset)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    original_save_dir = args.save_dir
    args.save_dir = f"{original_save_dir}_gpu{gpu_id + seed_offset}"

    from misc.utils import Logger
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)

    _, train_loader, val_loader, nclass = load_data(args, detailed=False)

    best_acc_l = []
    acc_l = []

    for i in range(args.repeat):
        plotter = Plotter(args.save_dir, args.epochs, idx=i)
        model = define_model(args, nclass, logger)

        best_acc, acc = train(args, model, train_loader, val_loader, plotter, logger, gpu_id)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

    final_best_acc = np.mean(best_acc_l)

    return_dict[gpu_id] = final_best_acc
    args.save_dir = original_save_dir


def main(args, logger, repeat=1):
    num_gpus = torch.cuda.device_count()

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id in range(num_gpus):
        args_copy = type(args)()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(args_copy, attr, getattr(args, attr))

        p = mp.Process(target=train_on_gpu, args=(gpu_id, args_copy, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    accuracies = []
    for gpu_id in range(num_gpus):
        if gpu_id in return_dict and return_dict[gpu_id] != 0.0:
            accuracies.append(return_dict[gpu_id])

    mean_all = 0.0
    std_all = 0.0

    if accuracies:
        mean_all = np.mean(accuracies)
        std_all = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

    acc_str = ", ".join([f"{acc:.1f}" for acc in accuracies])
    final_result_str = f"{acc_str} [{mean_all:.2f}±{std_all:.2f}]"

    print(f"\n{'=' * 50}")
    print(f"n_neighbors: {args.n_neighbors} size_min: {args.min_cluster_size} : {final_result_str}")

    log_file_path = os.path.join(args.save_dir, f'hard_log_{args.i_new}.txt')
    with open(log_file_path, 'w') as f:
        f.write(f"{final_result_str}")

    return accuracies


def train(args, model, train_loader, val_loader, plotter=None, logger=None, gpu_id=0):
    criterion = nn.CrossEntropyLoss().to(f"cuda:{gpu_id}")
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)

    model = model.to(f"cuda:{gpu_id}")

    for epoch in tqdm(range(cur_epoch + 1, args.epochs + 1), desc=f"GPU {gpu_id}", position=gpu_id):
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch,
                                          logger,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

            if plotter != None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None and args.verbose == True:
                    logger(f'GPU {gpu_id} - Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                logger=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()


    accumulation_steps = args.accumulation_steps

    optimizer.zero_grad()


    end = time.time()
    num_exp = 0
    device = next(model.parameters()).device
    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            input = input.to(device)
            target = target.to(device)

        data_time.update(time.time() - end)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None) and args.verbose == True:
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def validate(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None and args.verbose == True:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
            path, cur_epoch, checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, state, is_best):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    print("checkpoint saved! ", ckpt_path)


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args
    import multiprocessing as mp
    import torch.multiprocessing as torch_mp
    torch_mp.set_start_method('spawn', force=True)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)

    main(args, logger, args.repeat)
