import torch


class CFG:
    dataset_path = "../Shopee_Kaggle/folds.csv"
    image_folder = "../Shopee_Kaggle/train_images"
    
    DIM = (224, 224)

    num_workers = 16
    train_batch_size = 4
    valid_batch_size = 4
    epochs = 100
    seed = 2020
    lr = 1e-3

    device = torch.device('cuda')


    ### MODEL ###
    model_name = 'efficientnet_b0'  # model from timm, check if it preserves in models.py

    ### Metric Loss and its params ###
    loss_module = 'arcface'
    s = 30.0
    m = 0.5
    ls_eps = 0.0
    easy_margin = False


    ### Scheduler and its params ###
    SCHEDULER = 'CosineAnnealingWarmRestarts'
    factor = 0.2  # ReduceLROnPlateau
    patience = 3  # ReduceLROnPlateau
    eps = 1e-6  # ReduceLROnPlateau
    T_max = 10  # CosineAnnealingLR
    T_0 = 7  # CosineAnnealingWarmRestarts
    min_lr = 1e-6

    ### Model Params ###
    model_params = {
        'n_classes': 11014,
        'model_name': model_name,
        'use_fc': True,
        'fc_dim': 512,
        'dropout': 0.0,
        'loss_module': 'arcface',
        's': 30.0,
        'margin': 0.50,
        'ls_eps': 0.0,
        'pretrained': True
    }

