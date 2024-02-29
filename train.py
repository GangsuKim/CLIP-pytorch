import argparse

import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from tqdm import tqdm

from CLIP import CLIP
from CLIPDataSet import CLIPDataSet


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

            logits = model(images, label)  # logits = np.dot(I_e, T_e.T) * np.exp(t)

            # symmetric loss function
            matrix_labels = torch.eye(images.size(0)).to(device)  # labels = np.arange(e)
            loss_i = criterion(logits, matrix_labels)  # loss_i = cross_entropy_loss(logits, labels, axis=0)
            loss_t = criterion(logits.T, matrix_labels)  # loss_t = cross_entropy_loss(logits, labels, axis=1)
            loss = (loss_i + loss_t) / 2  # loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()

            train_loss[e] += loss.item()
            train_tq.set_description(f'Train_loss : {train_loss[e] / cnt}')

        train_loss[e] /= len(train_loader)

        torch.save(model.state_dict(), f'./weight/CLIP_E{e+1}.pth')
