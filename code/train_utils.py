import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import utils
from dataset import ShopeeDataset
from augmentations import get_train_transforms, get_valid_transforms
from config import CFG
from models import ShopeeNet


def run(data):
    train, valid = train_test_split(data, test_size=0.05, random_state=42)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    
    # Defining DataSet
    train_dataset = ShopeeDataset(
        csv=train,
        transforms=get_train_transforms(),
    )
        
    valid_dataset = ShopeeDataset(
        csv=valid,
        transforms=get_valid_transforms(),
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=CFG.num_workers
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = ShopeeNet(**CFG.model_params)
    model.to(CFG.device)
    parallel_model = torch.nn.DataParallel(model)
    
#     for name, param in model.named_parameters():
#         if name.startswith('backbone'):
#             param.requires_grad = False 
    
    
    # Defining criterion
    criterion = utils.fetch_loss()
    criterion.to(CFG.device)
        
    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(parallel_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]  
    
    optimizer = torch.optim.Adam(optimizer_parameters, lr=CFG.lr)
    # Defining LR Scheduler
    scheduler = utils.fetch_scheduler(optimizer)
        
    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(CFG.epochs):
        _ = train_fn(
            train_loader, parallel_model,
            criterion, optimizer,
            scheduler=scheduler,
            epoch=epoch
        )
        
        valid_loss = eval_fn(
            valid_loader,
            parallel_model,
            criterion,
            scheduler=scheduler
        )

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(parallel_model.module.state_dict(), f'{CFG.model_name}_{CFG.loss_module}_{best_loss:.3f}.bin')
            print('best model found for epoch {}'.format(epoch))


def train_fn(dataloader, model, criterion, optimizer, scheduler, epoch):
    model.train()
    loss_score = utils.AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(CFG.device)
        targets = targets.to(CFG.device)

        optimizer.zero_grad()

        output = model(images, targets)

        loss = criterion(output, targets)

        loss.mean().backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

    if scheduler is not None:
        scheduler.step()

    return loss_score


def eval_fn(data_loader, model, criterion, scheduler):
    loss_score = utils.AverageMeter()
    model.eval()

    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            image = d[0]
            targets = d[1]

            image = image.to(CFG.device)
            targets = targets.to(CFG.device)

            output = model(image, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.mean().detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    #         if scheduler is not None:
    #             scheduler.step(loss.detach().item())

    return loss_score
