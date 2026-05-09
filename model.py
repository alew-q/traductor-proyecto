import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):  # 512 → 128
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_len=128,  # 512 → 128
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_tgt_mask(self, size, device):
        return torch.triu(
            torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1
        )

    def encode(self, src, src_key_padding_mask):
        x = self.src_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.transformer.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        memory = self.encode(src, src_key_padding_mask)
        tgt_mask = self.make_tgt_mask(tgt.size(1), src.device)
        out = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)