from collections import Counter
import pandas as pd
import torch

__all__ = ["Vocabulary", "getEmbeddingMatrix"]

class Vocabulary():
    def __init__(self, path):
        df = pd.read_csv(path)
        self.vocab = Counter()
        for sentence in df["source"]:
            self.vocab.update(sentence.lower().split())
        self.vocab2idx={}
        self.idx2vocab=[]
        self.pad_token = "<pad>"
        self.unknown_token = "<unk>"
        self.start_token = "<sos>"
        self.end_token = "<eos>"
        self.pad_idx=0
        self.unknown_idx=1
        self.start_idx=2
        self.end_idx=3
        self.vocab2idx[self.pad_token]=0
        self.vocab2idx[self.unknown_token]=1
        self.vocab2idx[self.start_token]=2
        self.vocab2idx[self.end_token]=3
        self.idx2vocab.append(self.pad_token)
        self.idx2vocab.append(self.unknown_token)
        self.idx2vocab.append(self.start_token)
        self.idx2vocab.append(self.end_token)
        for token, freq in self.vocab.most_common():
            if freq < 2:
                break
            self.vocab2idx[token]=len(self.vocab2idx)
            self.idx2vocab.append(token)
        return

    def __len__(self):
        return len(self.idx2vocab)

    def get_idx(self, token):
        if token not in self.vocab2idx.keys():
            return self.unknown_idx
        return self.vocab2idx[token]

    def get_token(self, idx):
        return self.idx2vocab[idx]

    def encode_sequence(self, tokens):
        return [self.get_idx(token) for token in tokens]

    def decode_sequence(self, indices):
        return [self.get_token(idx) for idx in indices]

def getEmbeddingMatrix(vocabulary, path, embedding_dim=100, device=torch.device('cpu')):
    glove = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.split()
            glove[line[0]] = torch.tensor([float(x) for x in line[1:]], dtype=torch.float32, device=device)
    embedding_matrix = torch.randn((len(vocabulary), embedding_dim), device=device)*0.1
    for idx in range(len(vocabulary)):
        token = vocabulary.get_token(idx)
        embedding = glove.get(token)
        if embedding is not None:
            embedding_matrix[idx] = embedding
    return embedding_matrix

if __name__ =="__main__":
    vocabulary=Vocabulary("../dataset/Hindi_train.csv")
    torch.save(vocabulary, "../vocabulary.pt")
    print("vocabulary size: ", len(vocabulary))
    M = getEmbeddingMatrix(vocabulary, path="../glove.6B/glove.6B.100d.txt")
