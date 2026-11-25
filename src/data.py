import pandas as pd
from transformers import PreTrainedTokenizerFast, BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn

class training_dataset(Dataset):
    def __init__(self, path, tgt_tokenizer, vocabulary=None, pretrained=False):
        self.src_tokenizer=None
        self.pretrained=pretrained
        if pretrained:
            self.src_tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
        self.vocabulary=vocabulary
        df = pd.read_csv(path)
        if "translated" not in df.columns:
            df["translated"] = ""
        if not self.pretrained:
            self.X = df["source"].apply(lambda sentence: vocabulary.encode_sequence(sentence.split()))
        else:
            self.X = df["source"].apply(lambda sentence: self.src_tokenizer(sentence, truncation=True, add_special_tokens=True))
        self.tgt_tokenizer=tgt_tokenizer
        self.y = df["translated"].apply(lambda sentence: [self.tgt_tokenizer.bos_token_id]+self.tgt_tokenizer(sentence, truncation=True, add_special_tokens=False)['input_ids']+[self.tgt_tokenizer.eos_token_id])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.pretrained:
            return {
                "X": self.X[idx]["input_ids"],
                "X_mask": self.X[idx]["attention_mask"],
                "y": self.y[idx]
            }
        return {
            "X": self.X[idx],
            "y": self.y[idx]
        }

def collate_fn(batch, X_pad_idx=0, y_pad_idx=0, pretrained=False):
    X, y = [torch.tensor(sample["X"], dtype=torch.long) for sample in batch], [torch.tensor(sample["y"], dtype=torch.long) for sample in batch]
    if pretrained:
        X_mask = [torch.tensor(sample["X_mask"], dtype=torch.long) for sample in batch]
    X_length, y_length = torch.tensor([len(sample["X"]) for sample in batch]), torch.tensor([len(sample["X"]) for sample in batch])
    X, y = rnn.pad_sequence(X, batch_first=True, padding_value=X_pad_idx), rnn.pad_sequence(y, batch_first=True, padding_value=y_pad_idx)
    if pretrained:
        X_mask = rnn.pad_sequence(X_mask, batch_first=True, padding_value=0)
    return {
        "X": X,
        "X_length": X_length if not pretrained else X_mask,
        "y": y,
        "y_length": y_length
    }

if __name__=="__main__":
    import sys
    tgt_tokenizer=PreTrainedTokenizerFast.from_pretrained("../.tokenizer/Hindi")
    train_dataset=training_dataset("../dataset/Hindi_train.csv", tgt_tokenizer, pretrained=True)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=lambda batch: collate_fn(batch, train_dataset.src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id, pretrained=True))
    for batch in train_loader:
        y=batch["y"]
        break
    print("worked")
    sys.exit() 
    from vocabulary import Vocabulary
    vocabulary = Vocabulary("../dataset/Hindi_train.csv")
    tgt_tokenizer=PreTrainedTokenizerFast.from_pretrained("../.tokenizer/Hindi")
    print(f"translated vocabulary size: {tgt_tokenizer.vocab_size}")
    train_dataset = training_dataset("../dataset/Hindi_train.csv", tgt_tokenizer, vocabulary)
    print(train_dataset[1]["y"][0])
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=lambda batch: collate_fn(batch, vocabulary.pad_idx, tgt_tokenizer.pad_token_id))
    for batch in train_loader:
        y = batch["y"]
        print(y[:, 0])
        print(y[:, 1])
        break
    print("Script Ended")
