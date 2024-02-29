import torch.nn.functional as F
from ViT import ViT
import torch.nn as nn
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch


class CLIP(nn.Module):
    def __init__(self, tokenizer: GPT2Tokenizer, n_embedding_space: int = 512, temperature: float = 0.07, n_vocab: int = 50257):
        super(CLIP, self).__init__()

        # Text Encoder
        self.bpe = tokenizer
        self.text_encoder_config = GPT2Config(vocab_size=n_vocab, n_positions=76, n_embd=512, n_layer=12, n_head=8, output_hidden_states=True)
        self.text_encoder = GPT2LMHeadModel(self.text_encoder_config)
        self.W_t = nn.Linear(self.text_encoder_config.n_embd, n_embedding_space)

        # Image Encoder
        self.image_encoder = ViT(image_size=224, patch_size=16, dim=768, depth=12, heads=8, mlp_dim=3072)
        self.W_i = nn.Linear(self.image_encoder.dim, n_embedding_space)

        self.norm = nn.LayerNorm(normalized_shape=n_embedding_space)

        # Temperature Parameter
        self.temperature = temperature
        self.t = nn.Parameter(torch.FloatTensor([self.temperature]))

    def forward(self, image, texts):
        # extract feature representations of each modality

        # Text Tokenizing
        text_inputs = self.bpe(texts, padding=True, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        # Get [EOS] token's index from text token embeddings
        eos_places = torch.argmin(text_inputs.attention_mask, dim=1)
        eos_places[eos_places == 0] = len(text_inputs.attention_mask[-1])
        eos_places -= 1

        h = self.text_encoder(**text_inputs).hidden_states[-1]  # T_f = text_encoder(T) - 1
        T_f = h[torch.arange(h.size(0)), eos_places, :]  # T_f = text_encoder(T) - 2
        I_f = self.image_encoder(image)  # I_f = image_encoder(I)

        # joint multimodal embedding
        T_e = F.normalize(self.W_t(T_f), dim=1)  # T_e = l2_normalizer(np.dot(T_f, W_t), axi=1)
        I_e = F.normalize(self.W_i(I_f), dim=1)  # I_e = l2_normalizer(np.dot(I_f, W_i), axi=1)

        # scaled pairwise cosine similarities [n, n]
        logit = torch.matmul(I_e, T_e.T) * torch.exp(self.t)  # logits = np.dot(I_e, T_e.T) * np.exp(t)
        return logit
