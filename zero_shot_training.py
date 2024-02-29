import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CLIP import CLIP
from CLIPDataSet import CLIPZeroShotDataSet
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def main():
    # ================ [ArgumentParser] ================
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', default=None, help="Model checkpoint path")

    parser.add_argument('--load_first', action='store_true', help="if true, images are loaded to memory at loder")  # Inference data type
    parser.add_argument('--data_path', default='D:/AnacondaDrive/CLIP/data/archive/images/', type=str)
    parser.add_argument('--epoch_from', type=int, default=0, help="Last epoch of model")
    parser.add_argument('--n_classes', type=int, default=101, help="Number of the classification classes")

    # ================ [Train Settings] ================
    parser.add_argument('--lr', default=0.0005, type=float, help="model learning rate")
    args, unknown = parser.parse_known_args()
    print(args)

    train_df = pd.read_csv('./DATA/food101.csv', encoding='utf-8')

    n_epochs = 100

    train_dataset = CLIPZeroShotDataSet(train_df, origin_file_path=args.data_path, load_first=args.load_first, return_image=False, prefix_phrase='A photo of a *.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = torch.zeros(n_epochs)
    train_acc = torch.zeros(n_epochs)

    model = CLIP(tokenizer=train_dataset.bpe, n_embedding_space=512, n_vocab=train_dataset.n_vocab())

    # Load Self-supervised pretrained mdoel
    model.load_state_dict(torch.load('./weight/CLIP_E67.pth'))

    # Freeze Every Layers
    for param in model.parameters():
        param.requires_grad = False

    model.W_t = nn.Linear(model.text_encoder_config.n_embd, args.n_classes)

    model.to(device)

    # text_embedding_convertor = nn.Linear(model.text_encoder_config.n_embd, args.n_classes)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=0.000001)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epoch_from, n_epochs):
        model.eval()
        model.W_t.train()

        train_tq = tqdm(train_loader)

        cnt = 0
        for texts, labels in train_tq:
            labels = labels.to(device)
            cnt += 1
            # Text Token Embedding
            optimizer.zero_grad()
            text_inputs = model.bpe(texts, padding=True, return_tensors="pt").to(device)

            eos_places = torch.argmin(text_inputs.attention_mask, dim=1)
            eos_places[eos_places == 0] = len(text_inputs.attention_mask[-1])
            eos_places -= 1

            h = model.text_encoder(**text_inputs).hidden_states[-1]
            T_f = h[torch.arange(h.size(0)), eos_places, :]

            logit = model.W_t(T_f)
            loss = criterion(logit, labels)
            train_acc[e] += torch.sum(F.softmax(logit, dim=1).argmax(1) == labels).item()

            loss.backward()
            optimizer.step()

            train_loss[e] += loss.item()
            train_tq.set_description(f'Train_loss : {train_loss[e] / cnt}')

        train_loss[e] /= len(train_loader)
        train_acc[e] /= train_dataset.__len__()

        print(f'[Epoch {e}] Train Loss : {train_loss[e]}, Train acc : {train_acc[e]}')

        torch.save(model.state_dict(), f'./weight/CLIP_ZS_E{e + 1}.pth')


if __name__ == '__main__':
    main()
