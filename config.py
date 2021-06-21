class CFG:
    DIM = (224, 224)

    NUM_WORKERS = 16
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 100
    SEED = 2020
    LR = 1e-3

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
        'theta_zero': 0.785,
        'pretrained': True
    }

