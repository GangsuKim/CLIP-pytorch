import argparse

import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

from ViT import ViT

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image


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


class CLIP(nn.Module):
    def __init__(self, tokenizer: GPT2Tokenizer, n_embedding_space: int = 512, temperature: float = 0.07, n_vocab: int = 50257):
        super(CLIP, self).__init__()

        # Text Encoder
        self.bpe = tokenizer
        self.text_encoder_config = GPT2Config(vocab_size=n_vocab, n_positions=76, n_embd=512, n_layer=12, n_head=8, output_hidden_states=True)
        self.text_encoder = GPT2LMHeadModel(self.text_encoder_config)
        self.W_t = nn.Linear(self.text_encoder_config.n_embd, n_embedding_space)

        # Image Encoder
        self.image_encoder =ViT(image_size=224, patch_size=16, dim=768, depth=12, heads=8, mlp_dim=3072, dropout=0.0, emb_dropout=0.0)
        self.W_i = nn.Linear(self.image_encoder.dim, n_embedding_space)

        self.temperature = temperature
        self.t = nn.Parameter(torch.FloatTensor([self.temperature]))

    def forward(self, image, texts):
        # extract feature representations of each modality
        text_inputs = self.bpe(texts, padding=True, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        eos_places = torch.argmin(text_inputs.attention_mask, dim=1)
        eos_places[eos_places == 0] = len(text_inputs.attention_mask[-1])
        eos_places -= 1

        h = self.text_encoder(**text_inputs).hidden_states[-1]
        T_f = h[torch.arange(h.size(0)), eos_places, :]
        I_f = self.image_encoder(image)

        # joint multimodal embedding
        T_e = F.normalize(self.W_t(T_f), dim=1)
        I_e = F.normalize(self.W_i(I_f), dim=1)

        # scaled pairwise cosine similarities [n, n]
        logit = torch.matmul(I_e, T_e.T) * torch.exp(self.t)
        return logit


if __name__ == '__main__':
    # ================ [ArgumentParser] ================
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', default=None, help="Model checkpoint path")

    parser.add_argument('--load_first', action='store_true', help="if true, images are loaded to memory at loder")  # Inference data type
    parser.add_argument('--data_path', default='./data/flickr30k_images/', type=str)
    parser.add_argument('--epoch_from', type=int, default=0, help="Last epoch of model")

    # ================ [Train Settings] ================
    parser.add_argument('--lr', default=0.0005, help="model learning rate")
    args, unknown = parser.parse_known_args()
    print(args)

    # Load Data
    train_df = pd.read_csv('./DATA/train_3.csv', encoding='utf-8')

    n_epochs = 100
    batch_size = args.batch_size

    train_dataset = CLIPDataSet(train_df, origin_file_path=args.data_path, load_first=args.load_first)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = torch.zeros(n_epochs)

    model = CLIP(tokenizer=train_dataset.bpe, n_embedding_space=512, n_vocab=train_dataset.n_vocab())
    model.to(device)

    if args.checkpoint is not None:
        assert args.epoch_from > 1
        model.load_state_dict(torch.load(args.checkpoint))

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=0.000001, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epoch_from, n_epochs):
        model.train()

        train_tq = tqdm(train_loader)

        cnt = 0
        for images, label in train_tq:
            cnt += 1
            optimizer.zero_grad()

            images = images.to(device)

            logits = model(images, label)

            matrix_labels = torch.eye(images.size(0)).to(device)

            loss_i = criterion(logits, matrix_labels)
            loss_t = criterion(logits.T, matrix_labels)

            loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()

            train_loss[e] += loss.item()
            train_tq.set_description(f'Train_loss : {train_loss[e] / cnt}')

        train_loss[e] /= len(train_loader)

        torch.save(model.state_dict(), f'./weight/CLIP_E{e+1}.pth')
