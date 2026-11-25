import torch
from torch import nn
from src import training_dataset, collate_fn
from src import encoder, no_attention, pretrained_encoder, decoder, attention, seq2seq
from src import Vocabulary, getEmbeddingMatrix
from transformers import PreTrainedTokenizerFast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str)
parser.add_argument("--val_data_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model_number", type=int)
parser.add_argument("--beam_width", type=int)
parser.add_argument("--pretrained", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--dynamic_teacher_forcing", action="store_true", default=False)
parser.add_argument("--bpe_path", type=str, default="./.tokenizer/Hindi")
args = parser.parse_args()

MAX_LENGTH = 300
GLOVE_PATH = "./glove.6B/glove.6B.100d.txt"
BPE_PATH = args.bpe_path

BATCH_SIZE=64
EPOCHS=50
PATIENCE=3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocabulary=Vocabulary(args.train_data_path)
EmbeddingMatrix = getEmbeddingMatrix(vocabulary, GLOVE_PATH, embedding_dim=100, device=device)
tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(BPE_PATH)

train_dataset=None
if args.model_number == 1 or args.model_number == 2:
    train_dataset = training_dataset(args.train_data_path, tgt_tokenizer, vocabulary)
else:
    train_dataset = training_dataset(args.train_data_path, tgt_tokenizer, pretrained=True)
val_dataset=None
if args.model_number == 1 or args.model_number == 2:
    val_dataset = training_dataset(args.val_data_path, tgt_tokenizer, vocabulary)
else:
    val_dataset = training_dataset(args.val_data_path, tgt_tokenizer, pretrained=True)

src_pad_idx = vocabulary.pad_idx if vocabulary is not None else train_dataset.src_tokenizer.pad_token_id

from torch.utils.data import DataLoader
train_loader=None
if args.model_number == 1 or args.model_number == 2:
    train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id, pretrained=True), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader=None
if args.model_number == 1 or args.model_number == 2:
    val_loader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
else:
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, src_pad_idx, tgt_tokenizer.pad_token_id, pretrained=True), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

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

model = models[args.model_number].to(device)
if args.pretrained:
    model = torch.load(args.model_path, map_location=device)
optimizer = None
if args.model_number == 1 or args.model_number == 2:
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
elif args.model_number == 3:
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
else:
    optimizer=torch.optim.Adam([
        {"params": list(set(model.parameters())-set(model.encoder.rnn.parameters())), "lr": 1e-3},
        {"params": model.encoder.rnn.parameters(), "lr": 2e-5}
    ], weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)

def training_epoch(loader, model, optimizer, loss_fn, device, teacher_forcing_ratio=0.6):
    model.train()
    count=0
    for batch in loader:
        X, X_length, y, y_length = batch["X"].to(device, non_blocking=True), batch["X_length"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True), batch["y_length"].to(device, non_blocking=True)
        logits = model(X, X_length, y, y_length, pad_idx=src_pad_idx, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        count+=loss.item()
    history={
        "loss": count/len(loader)
    }
    return history

@torch.inference_mode()
def inference_loop(loader, model, loss_fn, device, teacher_forcing_ratio=0.6):
    model.eval()
    count=0
    count_real=0
    all_logits, all_labels=[], []
    for batch in loader:
        X, X_length, y, y_length = batch["X"].to(device, non_blocking=True), batch["X_length"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True), batch["y_length"].to(device, non_blocking=True)
        logits = model(X, X_length, y, y_length, pad_idx=src_pad_idx, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        count+=loss.item()
        logits = model(X, X_length, y, y_length, pad_idx=src_pad_idx, teacher_forcing_ratio=0.0)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        count_real+=loss.item()
        logits = model.greedy_decode(X, X_length, max_length=MAX_LENGTH, pad_idx=src_pad_idx)
        all_logits.append(list(torch.unbind(logits.detach().cpu(), dim=0)))
        all_labels.append(list(torch.unbind(y.detach().cpu(), dim=0)))
    history={
        "y_true": all_labels,
        "y_pred": all_logits,
        "loss": count/len(loader),
        "loss_real": count/len(loader)
    }
    return history

import math

best_score=1e3
res=0
import time
start_time=time.time()
for epoch in range(EPOCHS):
    if res==PATIENCE:
        print("---Patience Reached---")
        break
    training_history = training_epoch(train_loader, model, optimizer, loss_fn, device, teacher_forcing_ratio=0.6 if not args.dynamic_teacher_forcing else 8/(8+math.exp((epoch+1)/8)))
    training_loss=training_history["loss"]
    validation_history = inference_loop(val_loader, model, loss_fn, device, teacher_forcing_ratio=0.6 if not args.dynamic_teacher_forcing else 8/(8+math.exp((epoch+1)/8)))
    validation_loss=validation_history["loss"]
    torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, f"{args.checkpoint}/model/epoch_{epoch}.pt"
        )
    torch.save({
                "epoch": epoch,
                "training_history": training_history,
                "validation_history": validation_history
            }, f"{args.checkpoint}/result/epoch_{epoch}.pt"
        )
    if best_score>validation_loss:
        res=0
        best_score=validation_loss
        torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, f"{args.checkpoint}/model/epoch_{epoch}.pt"
        )
        torch.save({
                "epoch": epoch,
                "training_history": training_history,
                "validation_history": validation_history
            }, f"{args.checkpoint}/result/epoch_{epoch}.pt"
        )
    else:
        res+=1;
    print(f"EPOCH: {epoch} |  Training loss: {training_loss} | Validation loss: {validation_loss} | Ended at {time.time()-start_time}")
print(f"best validation loss obtained: {best_score}")
end_time=time.time()
print(f"Time taken for training: {end_time-start_time}")
