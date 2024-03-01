# CLIP PyTorch Implementation
Implementation of [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)  
[[Paper]](https://arxiv.org/abs/2103.00020) [[Official Code]](https://github.com/openai/CLIP)

## Architecture of CLIP
![CLIP](CLIP.png)  
Source : [OpenAI/CLIP](https://github.com/openai/CLIP)

## DataSet

- ### Pre-Training
  - [Flicker30K](https://shannon.cs.illinois.edu/DenotationGraph/)  
    - 31,783 data points
    - Only used 3rd label
  - [COCO 2015 Image Captioning Task](https://cocodataset.org/#captions-2015)  
    - 82,783 data points
    - Remove duplicated captions
- ### Zero-shot prediction
  - [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
    - 101 classes
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    - 10 classes

## Result
- **CLIP-Flick30K**
  - Training 211 epoch with Flickr30K
- **CLIP-Flick30K-MSCOCO**
  - Use pre-train as Flickr30K 200 epoch checkpoint
  - And train 7 epoch with COCO 2015 Image Captioning Task 


| DataSet  | CLIP-Flick30K |   CLIP-Flickr30K-COCO   |
|----------|:-------------:|:-----------------------:|
| Food101  | 1.1%      |          1.1%           |
| CIFAR-10 | 12.2%     |          16.8%          |