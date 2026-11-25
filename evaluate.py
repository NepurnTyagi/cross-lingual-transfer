import os
import torch
from transformers import PreTrainedTokenizerFast
import sacrebleu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(".tokenizer/Hindi")

for dr in os.listdir("./checkpoint"):
    training_loss, validation_loss = [], []
    bleu, chrf, ter = [], [], []
    for epoch in os.listdir(f"./checkpoint/{dr}/result"):
        y_true, y_pred  = [], []
        history = torch.load(f"./checkpoint/{dr}/result/{epoch}")
        training_loss.append(history["training_history"]["loss"])
        validation_loss.append(history["validation_history"]["loss"])
        for i in range(len(history["validation_history"]["y_true"])):
            unpadded = []
            for sample in history["validation_history"]["y_true"][i]:
                try:
                    idx = sample.tolist().index(tgt_tokenizer.eos_token_id)
                    unpadded.append(sample[1:idx])
                except ValueError:
                    unpadded.append(sample[1:])
            y_true.extend(tgt_tokenizer.batch_decode(unpadded))
            unpadded = []
            for sample in history["validation_history"]["y_pred"][i]:
                try:
                    idx = sample.tolist().index(tgt_tokenizer.eos_token_id)
                    unpadded.append(sample[1:idx])
                except ValueError:
                    unpadded.append(sample[1:])
            unpadded = tgt_tokenizer.batch_decode(unpadded)
            y_pred.extend([str(x) for x in unpadded])
        bleu.append(sacrebleu.corpus_bleu(y_pred, [y_true]).score) 
        chrf.append(sacrebleu.corpus_chrf(y_pred, [y_true]).score)
        ter.append(sacrebleu.corpus_ter(y_pred, [y_true]).score)
    fig, ax = plt.subplots()
    if dr not in ["model_45"]:
        ax.plot(np.arange(len(training_loss)), training_loss, color='r', label='Training Loss')
        ax.plot(np.arange(len(validation_loss)), validation_loss, color='y', label='Validation Loss')
    else:
        ax.plot(8/(8+np.exp(np.arange(len(training_loss))/8)), training_loss, color='r', label='Training Loss')
        ax.plot(8/(8+np.exp(np.arange(len(validation_loss))/8)), validation_loss, color='y', label='Validation Loss')
    ax.set_title(f"Loss Curve for {dr}")
    if dr not in ["model_45"]:
        ax.set_xlabel("Number Of Epochs")
    else:
        ax.set_xlabel("Teacher Forcing Ratio")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(f"results/{dr}_loss.png")
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(bleu)), bleu, color='b', label='bleu')
    ax.plot(np.arange(len(chrf)), chrf, color='y', label='chrf')
    ax.plot(np.arange(len(ter)), ter, color='r', label='ter')
    ax.set_title(f"Metrices score curve for {dr} on dev set")
    ax.set_xlabel("Number Of Epochs")
    ax.set_ylabel("Score")
    ax.legend()
    fig.savefig(f"results/{dr}_scores.png")
    plt.show()
import sys
sys.exit()

df = {
    "Model type": ["model_1", "model_2", "model_3", "model_4", "model_4", "model 4", "model 4", "model 4", "model 4 with dynamic teaacher forcing", "transfer learning with pretrained", "transfer learning with finetuning"],
    "BLUE": [],
    "chrF": [],
    "TER": []
    }
y_true = pd.read_csv("dataset/Hindi_public_test.csv")["translated"].fillna("").astype(str).tolist()
for dr in [1, 2, 3, 4]:
    y_pred = pd.read_csv(f"predictions/model_{dr}.csv")["translated"].fillna("").astype(str).tolist()
    df["BLUE"].append(sacrebleu.corpus_bleu(y_pred, [y_true]).score)
    df["chrF"].append(sacrebleu.corpus_chrf(y_pred, [y_true]).score)
    df["TER"].append(sacrebleu.corpus_ter(y_pred, [y_true]).score)
for beam_size in [1, 5, 10, 20]:
    y_pred = pd.read_csv(f"predictions/model_4_{beam_size}.csv")["translated"].fillna("").astype(str).tolist()
    df["BLUE"].append(sacrebleu.corpus_bleu(y_pred, [y_true]).score)
    df["chrF"].append(sacrebleu.corpus_chrf(y_pred, [y_true]).score)
    df["TER"].append(sacrebleu.corpus_ter(y_pred, [y_true]).score)
for dr in [45, 460, 461]:
    y_pred = pd.read_csv(f"predictions/model_{dr}.csv")["translated"].fillna("").astype(str).tolist()
    if dr in [460, 461]:
        y_true = pd.read_csv("dataset/Marathi_public_test.csv")["translated"].fillna("").astype(str).tolist()
    df["BLUE"].append(sacrebleu.corpus_bleu(y_pred, [y_true]).score)
    df["chrF"].append(sacrebleu.corpus_chrf(y_pred, [y_true]).score)
    df["TER"].append(sacrebleu.corpus_ter(y_pred, [y_true]).score)
#print(pd.DataFrame(df).to_latex())

selected_indices = [0, 22, 39, 105, 138]
for idx in selected_indices:
    df = {"Different Model and Generations": ["Model 1", "Model 2", "Model 3", "Model 4", "Model 4 (beam width=1)", "Model 4 (beam width=5)", "Model 4 (beam width=10)", "Model 4 (beam width=20)"],
          "Generated": []}
    for dr in [1, 2, 3, 4]:
        df["Generated"].append(pd.read_csv(f"predictions/model_{dr}.csv")["translated"][idx])
    for beam_size in [1, 5, 10, 20]:
        df["Generated"].append(pd.read_csv(f"predictions/model_4_{beam_size}.csv")["translated"][idx])
    print(pd.read_csv(f"dataset/Hindi_public_test.csv")["source"][idx])
    print(pd.read_csv(f"dataset/Hindi_public_test.csv")["translated"][idx])
    print(pd.DataFrame(df).to_latex())
