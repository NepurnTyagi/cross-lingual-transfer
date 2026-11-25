import pandas as pd
import torch
from torch import nn
from src import training_dataset, collate_fn
from src import encoder, no_attention, pretrained_encoder, decoder, attention, seq2seq
from src import Vocabulary, getEmbeddingMatrix
from transformers import PreTrainedTokenizerFast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--mode", type=int)
parser.add_argument("--decoding_strategy", type=str)
parser.add_argument("--beam_width", type=int, default=5)
parser.add_argument("--bpe_path", type=str, default="./.tokenizer/Hindi")
parser.add_argument("--vocabulary_path", type=str, default="./vocabulary.pt")
args = parser.parse_args()

MAX_LENGTH = 300
BPE_PATH = args.bpe_path

BATCH_SIZE=8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocabulary=torch.load(args.vocabulary_path, map_location=device, weights_only=False)
EmbeddingMatrix = torch.randn((len(vocabulary), 100), device=device)*0.1
tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(BPE_PATH)

val_dataset=None
if args.mode == 1 or args.mode == 2:
    val_dataset = training_dataset(args.input, tgt_tokenizer, vocabulary)
else:
    val_dataset = training_dataset(args.input, tgt_tokenizer, pretrained=True)

src_pad_idx = vocabulary.pad_idx if args.mode == 1 or args.mode == 2 else val_dataset.src_tokenizer.pad_token_id

from torch.utils.data import DataLoader
val_loader=None
if args.mode == 1 or args.mode == 2:
    val_loader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id))
else:
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id, pretrained=True))

models = []
models.append(None)
models.append(
    seq2seq(
        encoder(embedding_dim=100, hidden_dim=96, embedding_matrix=EmbeddingMatrix, precomputed=False),
        decoder(context_dim=192, hidden_dim=192, embedding_dim=192, vocabulary_size=8000),
        no_attention(),
        8000,
        tgt_tokenizer.bos_token_id,
        tgt_tokenizer.eos_token_id,
        tgt_tokenizer.pad_token_id
    )
)
models.append(
    seq2seq(
        encoder(embedding_dim=100, hidden_dim=96, embedding_matrix=EmbeddingMatrix),
        decoder(context_dim=192, hidden_dim=192, embedding_dim=192, vocabulary_size=8000),
        attention(encoder_dim=192, decoder_dim=192),
        8000,
        tgt_tokenizer.bos_token_id,
        tgt_tokenizer.eos_token_id,
        tgt_tokenizer.pad_token_id
    )
)
models.append(
    seq2seq(
        pretrained_encoder(192, True),
        decoder(context_dim=768, hidden_dim=192, embedding_dim=192, vocabulary_size=8000),
        attention(encoder_dim=768, decoder_dim=192),
        8000,
        tgt_tokenizer.bos_token_id,
        tgt_tokenizer.eos_token_id,
        tgt_tokenizer.pad_token_id
    )
)
models.append(
    seq2seq(
        pretrained_encoder(192, False),
        decoder(context_dim=768, hidden_dim=192, embedding_dim=192, vocabulary_size=8000),
        attention(encoder_dim=768, decoder_dim=192),
        8000,
        tgt_tokenizer.bos_token_id,
        tgt_tokenizer.eos_token_id,
        tgt_tokenizer.pad_token_id
    )
)
model = models[args.mode].to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])

@torch.inference_mode()
def inference_loop(loader, model, device):
    model.eval()
    all_logits = []
    for batch in loader:
        X, X_length = batch["X"].to(device, non_blocking=True), batch["X_length"].to(device, non_blocking=True)
        logits = None
        if args.decoding_strategy == "greedy":
            logits=model.greedy_decode(X, X_length, max_length=MAX_LENGTH, pad_idx=src_pad_idx)
        else:
            logits=model.beam_search(X, X_length, args.beam_width, max_length=MAX_LENGTH, pad_idx=src_pad_idx)
        unpadded = []
        for sample in logits.tolist():
            try:
                idx = sample.index(tgt_tokenizer.eos_token_id)
                unpadded.append(sample[1:idx])
            except ValueError:
                unpadded.append(sample[1:])
        all_logits.extend(tgt_tokenizer.batch_decode(unpadded))
    return all_logits

import time
start_time=time.time()
y_pred = inference_loop(val_loader, model, device)
df = pd.read_csv(args.input)
df["translated"] = y_pred
df.to_csv(args.output)
end_time=time.time()
print(f"Average Time taken for Generating a sentence: {(end_time-start_time)/len(df)}")
