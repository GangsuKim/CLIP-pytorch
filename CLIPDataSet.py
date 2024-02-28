from torch.utils.data import Dataset

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from torchvision.transforms import transforms

import os
from PIL import Image
import pandas as pd
from tqdm import tqdm


class CLIPDataSet(Dataset):
    def __init__(self, train_data: pd.DataFrame, origin_file_path: str, load_first: bool = True):
        # Text Encoder
        self.bpe = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.bpe.pad_token = self.bpe.eos_token

        self.origin_file_path = origin_file_path
        self.load_first = load_first

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4441, 0.4212, 0.3847], std=[0.2613, 0.2547, 0.2656])
        ])

        # Initial tokenizing
        self.image_data = []
        self.text_data = []

        for d in tqdm(train_data.iloc, total=len(train_data)):
            if self.load_first:
                image_ = self.transform(Image.open(os.path.join(self.origin_file_path, d.image_name)))
                self.image_data.append(image_)
            else:
                self.image_data.append(os.path.join(self.origin_file_path, d.image_name))

            text_ = self.bpe.bos_token + d.comment + self.bpe.eos_token
            self.text_data.append(text_)

        assert len(self.image_data) == len(self.text_data)

    def n_vocab(self) -> int:
        return self.bpe.vocab_size

    def __getitem__(self, i):
        if self.load_first:
            image_ = self.image_data[i]
        else:
            image_ = self.transform(Image.open(os.path.join(self.image_data[i])))

        text_ = self.text_data[i]
        return image_, text_

    def __len__(self):
        return len(self.image_data)