import torch
from torch.utils.data import Dataset
import cv2


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, mode="train", max_length=None):
        self.dataframe = dataframe
        if mode != "test":
            self.targets = dataframe['label_code'].values
        texts = list(dataframe['title'].apply(lambda o: str(o)).values)
        self.encodings = tokenizer(texts,
                                   padding=True,
                                   truncation=True,
                                   max_length=max_length)
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        # when testing, there are no targets so we won't do the following
        if self.mode != "test":
            item['labels'] = torch.tensor(self.targets[idx]).long()
        return item


class ShopeeDataset(Dataset):
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        # text = row.title

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(row.label_group)
