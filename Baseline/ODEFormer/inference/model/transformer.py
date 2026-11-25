# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Source: https://github.com/sdascoli/odeformer

from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Baseline.ODEFormer.inference.model.embedders import TwoHotEmbedder

N_MAX_POSITIONS = 4096
logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None, use_two_hot=False):
    if use_two_hot:
        m = TwoHotEmbedder(num_embeddings, embedding_dim, padding_idx=padding_idx)
    else:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** (-0.5))
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return (mask, attn_mask)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None, use_cache=False):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (mask == 0).view(mask_reshape).expand_as(scores)
            scores.masked_fill_(mask, -float("inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, v)
        context = unshape(context)
        if TransformerModel.STORE_OUTPUTS and (not self.training):
            self.outputs = weights.detach().cpu()
        return self.out_lin(context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout, activation):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.activation = getattr(F, activation)
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = self.activation(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerModel(nn.Module):
    STORE_OUTPUTS = False

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.use_cross_attention = params.use_cross_attention
        self.activation = params.activation
        if self.is_decoder:
            self.use_two_hot = params.use_two_hot
        self.apex = params.nvidia_apex
        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"
        if positional_embeddings is None or positional_embeddings in ["alibi", "none"]:
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError
        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(
                self.n_words,
                self.dim,
                padding_idx=self.pad_index,
                use_two_hot=not is_encoder and params.use_two_hot,
            )
        else:
            self.embeddings = None
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        self.cache = None
        if self.with_output:
            assert not self.use_prior_embeddings
            self.proj = nn.Linear(self.dim, self.n_words, bias=True)
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
        """
        slen, bs = x.size()[:2]
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = (
                torch.arange(src_len.max(), dtype=torch.long, device=lengths.device)
                < src_len[:, None]
            )
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)
        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        if TransformerModel.STORE_OUTPUTS and (not self.training):
            self.outputs = []
        if not self.use_prior_embeddings:
            tensor = self.embeddings(x)
        else:
            tensor = x
        if not self.use_cross_attention and self.is_decoder and (src_enc is not None):
            src_enc = src_enc.mean(dim=1)
            tensor[:, 0, :] = src_enc
        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and (not self.training):
            self.outputs.append(tensor.detach().cpu())
        for i in range(self.n_layers):
            self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            if self.is_decoder and src_enc is not None and self.use_cross_attention:
                self.encoder_attn[i].cache = self.cache
                attn = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and (not self.training):
                self.outputs.append(tensor.detach().cpu())
        if use_cache:
            self.cache["slen"] += tensor.size(1)
        tensor = tensor.transpose(0, 1)
        return tensor

    def generate(
        self,
        src_enc,
        src_len,
        max_len=200,
        top_p=1.0,
        sample_temperature=None,
        seed=0,
        average_across_batch=False,
        env=None,
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """
        torch.random.default_generator.manual_seed(seed)
        if src_enc.is_cuda:
            torch.cuda.default_generators[src_enc.device.index].manual_seed(seed)
        bs = len(src_len)
        assert src_enc.size(0) == bs
        if self.use_two_hot:
            generated = src_len.new(max_len, bs).to(self.dtype)
        else:
            generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)
        two_hot_constant_masks = [torch.zeros_like(generated[0], dtype=bool)]
        self.cache = {"slen": 0}
        while cur_len < max_len:
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :].to(self.dtype)
            scores = self.proj(tensor)
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                if average_across_batch:
                    scores = scores.mean(dim=0, keepdim=True).expand(bs, scores.size(1))
                logits = F.softmax(scores.float() / sample_temperature, dim=1)
                next_words = torch.multinomial(logits, num_samples=1).squeeze(1)
            assert next_words.size() == (bs,)
            if self.use_two_hot:
                next_words, two_hot_constant_mask = env.topk_decode_two_hot(
                    logits=scores,
                    topk_idx=next_words,
                    unfinished_sents=unfinished_sents,
                )
                two_hot_constant_masks.append(two_hot_constant_mask)
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                1 - unfinished_sents
            )
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        two_hot_constant_masks = torch.stack(two_hot_constant_masks)
        return (generated[:cur_len], gen_len, two_hot_constant_masks)
