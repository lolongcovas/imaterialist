import os
from torch.utils.data import DataLoader
import numpy as np
import argparse
import models
from tensorboardX import SummaryWriter
import json
from dataset import dataset
import torch
import torch.nn as nn
import tensorboard_logger as tflogger


NUM_WORKERS = 8

if 'NUM_WORKERS' in os.environ:
    NUM_WORKERS = int(os.environ['NUM_WORKERS'])


def parse_json(filename):
    data = json.load(open(filename))
    MAX_LABEL = 228
    image_filenames = [x['url'].replace('https://', '')
                       .replace('http://', '') for x in data['images']]
    imageId2labels = {x['imageId']: map(int, x['labelId'])
                      for x in data['annotations']}
    image_labels = [np.array(imageId2labels[x['imageId']])-1
                    for x in data['images']]
    
    return image_filenames, image_labels


if __name__ == '__main__':

    model_names = [name for name in models.__dict__
                   if not name.startswith('__')
                   and callable(models.__dict__[name])]
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', help='model architecture'
                        + ' | '.join(model_names),
                        type=str)
    parser.add_argument('-weights', help='weights filename',
                        type=str)
    parser.add_argument('-epochs', help='total epochs. default 20',
                        type=int, default=20)
    parser.add_argument('-resume', help='epoch to resume from',
                        type=int, default=0)
    parser.add_argument('-lr', help='learning rate. default 0.1',
                        type=float, default=0.1)
    parser.add_argument('-bsize', help='batch size',
                        type=int, default=128)
    parser.add_argument('-log', help='log path',
                        type=str)
    parser.add_argument('-bpath', help='image path',
                        type=str,
                        default='/media/nas/private-dataset/imaterialist')
    parser.add_argument('-out', help='output model filename',
                        type=str)

    args = parser.parse_args()
    
    log_path = args.log
    if log_path is None:
        log_path = 'logs/%s' % (args.net)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    summary_writer = SummaryWriter(log_path)

    train_filenames, train_labels = parse_json('jsons/train.json')
    val_filenames, val_labels = parse_json('jsons/validation.json')
    train_dataset = dataset(args.bpath,
                            train_filenames, train_labels, validation=True)
    val_dataset = dataset(args.bpath,
                          val_filenames, val_labels,
                          validation=True)

    batch_size = args.bsize
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

    model = models.__dict__[args.net](nClasses=228)
    # model.set_fine_tune_level(args.ft, args.seed)
    if args.weights:
        weights = torch.load(args.weights)['state_dict']
        model.load_model(weights)
    torch.backends.cudnn.benchmark = True
    model = model.cuda()
    model = nn.DataParallel(model)

    # setup criterion
    criterion = nn.BCEWithLogitsLoss().cuda()

    lr = args.lr
    momentum = 0.99
    weight_decay = 1e-3
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr,
        momentum=momentum,
        weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=max(args.epochs // 3, 1),
                                                gamma=0.1)

    # sample_ratio = 0.975
    # label_hist = np.histogram(labels, bins=5)[0]
    # label_hist = 1 / (label_hist.astype(np.float32) / np.sum(label_hist))

    for epoch in range(args.resume, args.epochs):
        scheduler.step()
        # alpha = sample_ratio ** epoch
        # sampler_weights = alpha * w0 + (1 - alpha) * w1
        # weights = [sampler_weights[label] for label in tr_labels]

        # sampler = torch.utils.data.sampler.WeightedRandomSampler(
        #     weights,
        #     len(train_dataset))
    
        # train_dataloader = DataLoader(train_dataset,
        #                               batch_size=batch_size,
        #                               shuffle=sampler is None,
        #                               num_workers=NUM_WORKERS,
        #                               pin_memory=True,
        #                               sampler=sampler)
        model.train(True)
        mean = 0
        std = 0
        for i, (X, y) in enumerate(train_dataloader):
            X = torch.autograd.Variable(X.cuda())
            """
            mean += X.transpose(1, 0).contiguous().view(3, -1).mean(1)
            std += X.transpose(1, 0).contiguous().view(3, -1).std(1)
            continue
            """
            y = torch.autograd.Variable(y.cuda())
            y_pred = model(X)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                # visualize input images                
                tensor = X.data.cpu().numpy()\
                         * np.array([[0.2777, 0.1981, 0.1574]]).reshape([1, 3, 1, 1])\
                         + np.array([[0.3940, 0.2713, 0.1869]]).reshape([1, 3, 1, 1])
                tflogger.visualize_tensors(summary_writer,
                                           torch.from_numpy(tensor),
                                           'image',
                                           i + epoch * len(train_dataloader))
                # visualize the weights
                weights = [x for x in model.parameters()][0]\
                          .data.cpu().numpy().copy()
                tflogger.visualize_tensors(summary_writer,
                                           torch.from_numpy(weights),
                                           'conv1',
                                           i + epoch * len(train_dataloader))

                tflogger.visualize_histogram(summary_writer, model,
                                             i + epoch * len(train_dataloader))
                summary_writer.add_scalar('loss/train',
                                          loss.data.cpu().numpy(),
                                          i + epoch * len(train_dataloader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(val_dataloader):
                X = torch.autograd.Variable(X.cuda(),
                                            requires_grad=False)
                # mean += X.transpose(1, 0).contiguous().view(3, -1).mean(1)
                # std += X.transpose(1, 0).contiguous().view(3, -1).std(1)
                if args.objective == 'classification':
                    y = torch.autograd.Variable(y.cuda(),
                                                requires_grad=False)
                else:
                    y = torch.autograd.Variable(y.view(y.shape[0], 1)
                                                .type(torch.FloatTensor).cuda(),
                                                requires_grad=False,)
                y_pred = model(X)
                val_loss += criterion(y_pred, y)
        val_loss /= len(val_dataloader)
        summary_writer.add_scalar('loss/val', val_loss.data.cpu().numpy(),
                                  epoch * len(train_dataloader))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   '%s_epoch_%d.pth' % (args.out, epoch))


