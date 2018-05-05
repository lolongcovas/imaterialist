import os
import tqdm
import gzip
from torch.utils.data import DataLoader
import numpy as np
import argparse
import models
from tensorboardX import SummaryWriter
import json
from dataset import dataset
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import tensorboard_logger as tflogger


NUM_WORKERS = 8

if 'NUM_WORKERS' in os.environ:
    NUM_WORKERS = int(os.environ['NUM_WORKERS'])


def binary_cross_entropy_with_logits(input, target):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    return loss.mean()


def parse_json(filename):
    data = json.load(gzip.open(filename))
    MAX_LABEL = 228
    image_filenames = [x['url'].replace('https://', '')
                       .replace('http://', '') for x in data['images']]
    imageId2labels = {x['imageId']: map(int, x['labelId'])
                      for x in data['annotations']}
    image_labels = [np.array(imageId2labels[x['imageId']])-1
                    for x in data['images']]
    
    return image_filenames, image_labels


def find_best_f1_score(y, y_target):
    from sklearn.metrics import f1_score
    scores = np.linspace(0, 1, 100)
    best_f1 = 0
    thr = 0
    for score in scores:
        y_bin = y > score
        f1 = f1_score(y_target, y_bin)
        if f1 > best_f1:
            best_f1 = f1
            thr = score
    return best_f1, thr


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
                        type=int, default=80)
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

    train_filenames, train_labels = parse_json('jsons/train_filtered.json.gz')
    labels = []
    for label in train_labels:
        labels.extend(label.tolist())
    label_weights = np.histogram(labels, bins=228)[0].astype(np.float32)
    label_weights = np.sum(label_weights) / label_weights
    label_weights = label_weights / np.max(label_weights)
    samples_weight = [np.mean(label_weights[label]) for label in train_labels]

    val_filenames, val_labels = parse_json('jsons/validation.json.gz')
    train_dataset = dataset(args.bpath,
                            train_filenames, train_labels,
                            validation=False)
    val_dataset = dataset(args.bpath,
                          val_filenames, val_labels,
                          validation=True)

    batch_size = args.bsize
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  sampler=WeightedRandomSampler(samples_weight,
                                                                len(samples_weight)))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

    model = models.__dict__[args.net](nClasses=228)
    # model.set_fine_tune_level(args.ft, args.seed)
    model = model.cuda()
    # model = nn.DataParallel(model)
    if args.weights:
        weights = torch.load(args.weights)['state_dict']
        model.load_state_dict(weights)
        # model.load_model(weights)
    torch.backends.cudnn.benchmark = True

    model.set_fine_tune_level(level=3)

    # setup criterion
    criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(label_weights))\
                  .cuda()

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
        for i, (X, y) in enumerate(tqdm.tqdm(train_dataloader)):
            X = torch.autograd.Variable(X.cuda())
            """
            mean += X.transpose(1, 0).contiguous().view(3, -1).mean(1)
            std += X.transpose(1, 0).contiguous().view(3, -1).std(1)
            continue
            """
            y = torch.autograd.Variable(y.cuda())
            y_pred = model(X)
            optimizer.zero_grad()
            loss = criterion(y_pred, y) * 100
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                # visualize input images
                # tensor = X.data.cpu().numpy()\
                #          * np.array([[0.3363, 0.3329, 0.3268]]).reshape([1, 3, 1, 1])\
                #          + np.array([[0.5883, 0.5338, 0.5273]]).reshape([1, 3, 1, 1])
                # tflogger.visualize_tensors(summary_writer,
                #                            torch.from_numpy(tensor),
                #                            'image',
                #                            i + epoch * len(train_dataloader))
                # # visualize the weights
                # weights = [x for x in model.parameters()][0]\
                #           .data.cpu().numpy().copy()
                # tflogger.visualize_tensors(summary_writer,
                #                            torch.from_numpy(weights),
                #                            'conv1',
                #                            i + epoch * len(train_dataloader))

                # tflogger.visualize_histogram(summary_writer, model,
                #                              i + epoch * len(train_dataloader))
                summary_writer.add_scalar('loss/train',
                                          loss.data.cpu().numpy(),
                                          i + epoch * len(train_dataloader))
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   '%s_epoch_%d.pth' % (args.out, epoch))
    model.eval()
    val_loss = 0
    targets = np.zeros([0, 228])
    predictions = np.zeros([0, 228])
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm.tqdm(val_dataloader)):
            X = torch.autograd.Variable(X.cuda(),
                                        requires_grad=False)
            y = torch.autograd.Variable(y.cuda())
            y_pred = model(X)
            prob = torch.nn.functional.sigmoid(y_pred)
            predictions = np.vstack((predictions, prob.cpu().numpy()))
            targets = np.vstack((targets, y.cpu().numpy()))
    import epdb; epdb.set_trace()
    mean_f1 = []
    for cls in range(228):
        f1, thr = find_best_f1_score(predictions[:, cls],
                                     targets[:, cls])
        mean_f1.append(f1)
    np.mean(mean_f1)
