import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass


@dataclass
class Tacotron2Config:
    num_mels: int = 80

    # Character embeddings
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    # Encoder
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5

    # Decoder
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p: float = 0.1

    # Attention
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain="linear"):
        super().__init__()
        if padding is None:
            padding = "same"
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.num_chars, config.character_embed_dim, padding_idx=config.pad_token_id
        )
        self.convolutions = nn.ModuleList()
        for i in range(config.encoder_n_convolutions):
            in_ch = config.character_embed_dim if i == 0 else config.encoder_embed_dim
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(in_ch, config.encoder_embed_dim,
                             kernel_size=config.encoder_kernel_size,
                             stride=1, padding="same", dilation=1, w_init_gain="relu"),
                    nn.BatchNorm1d(config.encoder_embed_dim),
                    nn.ReLU(),
                    nn.Dropout(config.encoder_dropout_p),
                )
            )
        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim,
            hidden_size=config.encoder_embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths=None):
        x = self.embeddings(x).transpose(1, 2)   # (B, E, T)
        B, _, S = x.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), S, device=x.device)
        for block in self.convolutions:
            x = block(x)
        x = x.transpose(1, 2)                    # (B, T, E)
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)
        outputs, _ = self.lstm(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs


class Prenet(nn.Module):
    def __init__(self, input_dim, prenet_dim, prenet_depth, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        dims = [input_dim] + [prenet_dim] * prenet_depth
        self.layers = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(
                nn.Sequential(
                    LinearNorm(in_d, out_d, bias=False, w_init_gain="relu"),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout_p, training=True)
        return x


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        self.conv = ConvNorm(2, attention_n_filters,
                             kernel_size=attention_kernel_size, padding="same", bias=False)
        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights):
        x = self.conv(attention_weights).transpose(1, 2)
        return self.proj(x)


class LocalSensitiveAttention(nn.Module):
    def __init__(self, attention_dim, decoder_hidden_size, encoder_hidden_size,
                 attention_n_filters, attention_kernel_size):
        super().__init__()
        self.in_proj  = LinearNorm(decoder_hidden_size, attention_dim, bias=True,  w_init_gain="tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=False, w_init_gain="tanh")
        self.location  = LocationLayer(attention_n_filters, attention_kernel_size, attention_dim)
        self.energy_proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain="tanh")
        self.reset()

    def reset(self):
        self.enc_proj_cache = None

    def _energies(self, mel_input, encoder_output, cumulative_attn, mask):
        mel_proj = self.in_proj(mel_input).unsqueeze(1)
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)
        loc_feat = self.location(cumulative_attn)
        energies = self.energy_proj(
            torch.tanh(mel_proj + self.enc_proj_cache + loc_feat)
        ).squeeze(-1)
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        return energies

    def forward(self, mel_input, encoder_output, cumulative_attn, mask=None):
        energies = self._energies(mel_input, encoder_output, cumulative_attn, mask)
        attn_weights  = F.softmax(energies, dim=1)
        attn_context  = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)
        return attn_context, attn_weights


class PostNet(nn.Module):
    def __init__(self, num_mels, postnet_num_convs=5, postnet_n_filters=512,
                 postnet_kernel_size=5, postnet_dropout_p=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                ConvNorm(num_mels, postnet_n_filters, kernel_size=postnet_kernel_size,
                         padding="same", w_init_gain="tanh"),
                nn.BatchNorm1d(postnet_n_filters),
                nn.Tanh(),
                nn.Dropout(postnet_dropout_p),
            )
        )
        for _ in range(postnet_num_convs - 2):
            self.convs.append(
                nn.Sequential(
                    ConvNorm(postnet_n_filters, postnet_n_filters,
                             kernel_size=postnet_kernel_size, padding="same", w_init_gain="tanh"),
                    nn.BatchNorm1d(postnet_n_filters),
                    nn.Tanh(),
                    nn.Dropout(postnet_dropout_p),
                )
            )
        self.convs.append(
            nn.Sequential(
                ConvNorm(postnet_n_filters, num_mels,
                         kernel_size=postnet_kernel_size, padding="same"),
                nn.BatchNorm1d(num_mels),
                nn.Dropout(postnet_dropout_p),
            )
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.convs:
            x = block(x)
        return x.transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prenet = Prenet(
            config.num_mels, config.decoder_prenet_dim,
            config.decoder_prenet_depth, config.decoder_prenet_dropout_p,
        )
        self.rnn = nn.ModuleList([
            nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim,
                        config.decoder_embed_dim),
            nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim,
                        config.decoder_embed_dim),
        ])
        self.attention = LocalSensitiveAttention(
            attention_dim=config.attention_dim,
            decoder_hidden_size=config.decoder_embed_dim,
            encoder_hidden_size=config.encoder_embed_dim,
            attention_n_filters=config.attention_location_n_filters,
            attention_kernel_size=config.attention_location_kernel_size,
        )
        self.mel_proj  = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim,
                                    config.num_mels)
        self.stop_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim,
                                    1, w_init_gain="sigmoid")
        self.postnet = PostNet(
            num_mels=config.num_mels,
            postnet_num_convs=config.decoder_postnet_num_convs,
            postnet_n_filters=config.decoder_postnet_n_filters,
            postnet_kernel_size=config.decoder_postnet_kernel_size,
            postnet_dropout_p=config.decoder_postnet_dropout_p,
        )

    def _init_decoder(self, encoder_outputs, encoder_mask=None):
        B, S, E = encoder_outputs.shape
        dev = encoder_outputs.device
        self.h = [torch.zeros(B, self.config.decoder_embed_dim, device=dev) for _ in range(2)]
        self.c = [torch.zeros(B, self.config.decoder_embed_dim, device=dev) for _ in range(2)]
        self.cumulative_attn_weight = torch.zeros(B, S, device=dev)
        self.attn_weight   = torch.zeros(B, S, device=dev)
        self.attn_context  = torch.zeros(B, self.config.encoder_embed_dim, device=dev)
        self.encoder_outputs = encoder_outputs
        self.encoder_mask    = encoder_mask

    def _bos_frame(self, B):
        return torch.zeros(B, 1, self.config.num_mels)

    def decode(self, mel_step):
        rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)
        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))
        attn_hidden = F.dropout(self.h[0], self.config.attention_dropout_p, self.training)
        attn_weights_cat = torch.cat(
            [self.attn_weight.unsqueeze(1), self.cumulative_attn_weight.unsqueeze(1)], dim=1
        )
        attn_context, attn_weights = self.attention(
            attn_hidden, self.encoder_outputs, attn_weights_cat, mask=self.encoder_mask
        )
        self.attn_weight            = attn_weights
        self.cumulative_attn_weight = self.cumulative_attn_weight + attn_weights
        self.attn_context           = attn_context
        decoder_input = torch.cat([attn_hidden, self.attn_context], dim=-1)
        self.h[1], self.c[1] = self.rnn[1](decoder_input, (self.h[1], self.c[1]))
        dec_hidden    = F.dropout(self.h[1], self.config.decoder_dropout_p, self.training)
        proj_input    = torch.cat([dec_hidden, self.attn_context], dim=-1)
        mel_out       = self.mel_proj(proj_input)
        stop_out      = self.stop_proj(proj_input)
        return mel_out, stop_out, attn_weights

    def forward(self, encoder_outputs, encoder_mask, mels, decoder_mask):
        bos = self._bos_frame(mels.shape[0]).to(encoder_outputs.device)
        mels_w_start = torch.cat([bos, mels], dim=1)
        self._init_decoder(encoder_outputs, encoder_mask)
        mel_outs, stop_tokens, attention_weights = [], [], []
        T_dec   = mels.shape[1]
        mel_proj = self.prenet(mels_w_start)
        for t in range(T_dec):
            if t == 0:
                self.attention.reset()
            mel_out, stop_out, attn_w = self.decode(mel_proj[:, t, :])
            mel_outs.append(mel_out)
            stop_tokens.append(stop_out)
            attention_weights.append(attn_w)
        mel_outs         = torch.stack(mel_outs, dim=1)
        stop_tokens      = torch.stack(stop_tokens, dim=1).squeeze(-1)
        attention_weights = torch.stack(attention_weights, dim=1)
        mel_residual     = self.postnet(mel_outs)
        dec_mask = decoder_mask.unsqueeze(-1).bool()
        mel_outs          = mel_outs.masked_fill(dec_mask, 0.0)
        mel_residual      = mel_residual.masked_fill(dec_mask, 0.0)
        attention_weights = attention_weights.masked_fill(dec_mask, 0.0)
        stop_tokens       = stop_tokens.masked_fill(dec_mask.squeeze(-1), 1e3)
        return mel_outs, mel_residual, stop_tokens, attention_weights

    @torch.inference_mode()
    def inference(self, encoder_output, max_decode_steps=1000):
        bos = self._bos_frame(B=1).squeeze(0).to(encoder_output.device)
        self._init_decoder(encoder_output, encoder_mask=None)
        mel_outs, stop_outs, attn_weights = [], [], []
        _input = bos
        self.attention.reset()
        while True:
            _input = self.prenet(_input)
            mel_out, stop_out, attn_w = self.decode(_input)
            mel_outs.append(mel_out)
            stop_outs.append(stop_out)
            attn_weights.append(attn_w)
            if torch.sigmoid(stop_out).item() > 0.5:
                break
            if len(mel_outs) >= max_decode_steps:
                print("Reached max decoder steps")
                break
            _input = mel_out
        mel_outs    = torch.stack(mel_outs, dim=1)
        stop_outs   = torch.stack(stop_outs, dim=1).squeeze(-1)
        attn_weights = torch.stack(attn_weights, dim=1)
        mel_residual = self.postnet(mel_outs)
        return mel_outs, mel_residual, stop_outs, attn_weights


class Tacotron2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config  = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask):
        enc_out = self.encoder(text, input_lengths)
        mel_outs, mel_residual, stop_tokens, attn_weights = self.decoder(
            enc_out, encoder_mask, mels, decoder_mask
        )
        mel_postnet = mel_outs + mel_residual
        return mel_outs, mel_postnet, stop_tokens, attn_weights

    @torch.inference_mode()
    def inference(self, text, max_decode_steps=1000):
        if text.ndim == 1:
            text = text.unsqueeze(0)
        assert text.shape[0] == 1, "Inference batch size must be 1"
        enc_out = self.encoder(text)
        mel_outs, mel_residual, stop_outs, attn_weights = self.decoder.inference(
            enc_out, max_decode_steps=max_decode_steps
        )
        return mel_outs + mel_residual, attn_weights
