import pandas as pd
from glob import glob
import os


def food101(data_path: str):
    labels = {}
    data_dict = {'text':[], 'label':[], 'image_name':[]}

    for i, folder_name in enumerate(os.listdir(data_path)):
        labels[folder_name] = i

    for file in glob(os.path.join(data_path, '*/*.jpg')):
        temp = file.split('\\')
        data_dict['text'].append(temp[1].replace('_', ' '))
        data_dict['label'].append(labels[temp[1]])
        data_dict['image_name'].append(temp[2])

    return pd.DataFrame(data_dict)


if __name__ == '__main__':
    df = food101(data_path='D:/AnacondaDrive/CLIP/data/archive/images/')
    df.to_csv('D:/AnacondaDrive/CLIP/data/archive/food101.csv', encoding='utf-8', index_label=0)
