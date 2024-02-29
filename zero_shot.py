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
    parser.add_argument('--load_first', action='store_true', help="if true, images are loaded to memory at loder")  # Inference data type
    parser.add_argument('--data_path', default='./data/CIFAR-10/test/', type=str)
    parser.add_argument('--n_classes', type=int, default=101, help="Number of the classification classes")

    # ================ [Train Settings] ================
    args, unknown = parser.parse_known_args()
    print(args)

    train_df = pd.read_csv('./DATA/CIFAR-10.csv', encoding='utf-8')

    train_dataset = CLIPZeroShotDataSet(train_df, origin_file_path=args.data_path, load_first=args.load_first, prefix_phrase='a photo of *, a type of food.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_acc = torch.zeros(1)

    model = CLIP(tokenizer=train_dataset.bpe, n_embedding_space=512, n_vocab=train_dataset.n_vocab())

    # Load Self-supervised pretrained model
    model.load_state_dict(torch.load('./weight/CLIP_E211.pth'))

    model.to(device)

    for e in range(1):
        model.eval()

        texts = train_dataset.text_data

        # Text Embedding
        text_inputs = model.bpe(texts, padding=True, return_tensors="pt").to(device)

        eos_places = torch.argmin(text_inputs.attention_mask, dim=1)
        eos_places[eos_places == 0] = len(text_inputs.attention_mask[-1])
        eos_places -= 1

        h = model.text_encoder(**text_inputs).hidden_states[-1]
        T_f = h[torch.arange(h.size(0)), eos_places, :]
        T_e = F.normalize(model.W_t(T_f))

        train_tq = tqdm(train_loader)
        cnt = 0
        for images, labels in train_tq:
            labels = labels.to(device)
            images = images.to(device)

            cnt += len(images)

            I_e = F.normalize(model.W_i(model.image_encoder(images)))

            logit = torch.matmul(I_e, T_e.T) * torch.exp(model.t)
            train_acc[e] += torch.sum(F.softmax(logit, dim=1).argmax(1) == labels).item()

            train_tq.set_description(f'Train acc : {train_acc[e] / cnt}')

        train_acc[e] /= train_dataset.__len__()

        print(f'[Epoch {e}] Train acc : {train_acc[e]}')


if __name__ == '__main__':
    main()
