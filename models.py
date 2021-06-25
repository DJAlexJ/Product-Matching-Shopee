import torch
import torch.nn as nn
import timm


class BertModel(nn.Module):
    def __init__(self,
                 bert_model,
                 num_classes=NUM_CLASSES,
                 last_hidden_size=CFG.bert_hidden_size):
        super().__init__()
        self.bert_model = bert_model
        self.arc_margin = ArcMarginProduct(last_hidden_size,
                                           num_classes,
                                           s=30.0,
                                           m=0.50,
                                           easy_margin=False)

    def get_bert_features(self, batch):
        output = self.bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_hidden_state = output.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)
        CLS_token_state = last_hidden_state[:, 0, :]  # obtaining CLS token state which is the first token.
        return CLS_token_state

    def forward(self, batch):
        CLS_hidden_state = self.get_bert_features(batch)
        output = self.arc_margin(CLS_hidden_state, batch['labels'])
        return output


class ShopeeNet(nn.Module):
    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 pretrained=True):

        super(ShopeeNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = 1536  # default value
        if model_name.startswith('efficientnet'):
            print("EFFNET")
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'nfnet' in model_name:
            print("NFNET")
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head = nn.Identity()
        elif model_name.startswith('densenet'):
            print("DENSENET")
            final_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        self.backbone.global_pool = nn.Identity()

        # custom poolings
        #  self.rmac_pooling = RMAC()
        #  self.gem_pooling = GeM()
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn1 = nn.BatchNorm2d(final_in_features)
            self.bn2 = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'softmax':
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module == 'arcface':
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.pooling(x).view(batch_size, -1)
        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn2(x)

        x = F.normalize(x)

        return x


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
