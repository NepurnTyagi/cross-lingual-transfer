import torch
from torch import nn
from transformers import BertModel

class encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, embedding_matrix, pad_idx=0, precomputed=True, decoder_dim=192):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.pad_idx=pad_idx
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, padding_idx=pad_idx, freeze=True)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.precomputed=precomputed
        self.decoder_dim=decoder_dim
        self.layer=None
        if self.precomputed:
            self.layer = nn.Linear(2*hidden_dim, decoder_dim, bias=False)

    def forward(self, X, length, pad_idx):
        mask = X==pad_idx
        X = self.embedding(X)
        X = nn.utils.rnn.pack_padded_sequence(X, length.detach().cpu(), batch_first=True, enforce_sorted=False)
        X, (h, c) = self.rnn(X)
        X, _  = nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=self.pad_idx)
        h, c = torch.concat([h[0], h[1]], dim=1).unsqueeze(1), torch.concat([c[0], c[1]], dim=1).unsqueeze(1)
        precomputed_X=None
        if self.precomputed:
            precomputed_X = self.layer(X)
        return X, precomputed_X, (h, c), mask

class pretrained_encoder(nn.Module):
    def __init__(self, hidden_dim, freeze, precomputed=True, decoder_dim=192):
        super().__init__()
        self.rnn = BertModel.from_pretrained("bert-base-cased")
        if freeze:
            for param in self.rnn.parameters():
                param.requires_grad=False
        self.h = nn.Linear(self.rnn.config.hidden_size, hidden_dim)
        self.c = nn.Linear(self.rnn.config.hidden_size, hidden_dim)
        self.precomputed=precomputed
        self.layer=None
        if self.precomputed:
            self.layer = nn.Linear(self.rnn.config.hidden_size, decoder_dim, bias=False)

    def forward(self, X, X_length, pad_idx=None):
        X_mask = X_length
        X = self.rnn(input_ids=X, attention_mask=X_mask)
        X, hidden = X.last_hidden_state, X.pooler_output
        mask = X_mask==0
        h, c = torch.tanh(self.h(hidden)).unsqueeze(1), torch.tanh(self.c(hidden)).unsqueeze(1)
        precomputed_X=None
        if self.precomputed:
            precomputed_X=self.layer(X)
        return X, precomputed_X, (h, c), mask

class decoder(nn.Module):
    def __init__(self, context_dim, hidden_dim, embedding_dim, vocabulary_size):
        super().__init__()
        assert(hidden_dim==embedding_dim)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=context_dim+embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.out = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, context, y, hidden):
        h_n, c_n = hidden
        h_n, c_n = torch.permute(h_n, (1, 0, 2)), torch.permute(c_n, (1, 0, 2))
        hidden, (h_n, c_n) = self.rnn(torch.concat([context, self.embedding(y).unsqueeze(1)], dim=2), (h_n, c_n))
        y = self.out(hidden.squeeze(1))
        h_n, c_n = torch.permute(h_n, (1, 0, 2)), torch.permute(c_n, (1, 0, 2))
        return y, (h_n, c_n)

class no_attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_output, precomputed_X, hidden, mask=None):
        X_length = encoder_output.shape[1]-mask.sum(dim=1)
        batch_idx = torch.arange(encoder_output.shape[0], device=encoder_output.device)
        return torch.concat([encoder_output[batch_idx, X_length-1, :encoder_output.shape[2]], encoder_output[batch_idx, 0, encoder_output.shape[2]:]], dim=1).unsqueeze(1)

class attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.W=nn.Linear(decoder_dim, decoder_dim, bias=True)
        self.v = nn.Linear(decoder_dim, 1)

    def forward(self, encoder_output, precomputed_X, hidden, mask=None):
        a = self.v(nn.functional.tanh(precomputed_X+self.W(hidden.repeat(1, encoder_output.shape[1], 1)))).squeeze(-1)
        a = a.masked_fill(mask, -1e9)
        a = torch.softmax(a, dim=1)
        context = torch.bmm(a.unsqueeze(1), encoder_output)
        return context

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, attention, translated_vocabulary_size, sos_idx, eos_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.translated_vocabulary_size=translated_vocabulary_size
        self.sos_idx=sos_idx
        self.eos_idx=eos_idx
        self.pad_idx=pad_idx

    def forward(self, X, X_length, y, y_length, pad_idx, teacher_forcing_ratio=0.6):
        all_logits=torch.zeros(X.shape[0], y.shape[1]-1, self.translated_vocabulary_size, device=X.device)
        token_idx = y[:, 0]
        X, precomputed_X, (h, c), mask = self.encoder(X, X_length, pad_idx)
        for t in  range(1, y.shape[1]):
            context=self.attention(X, precomputed_X, h, mask=mask)
            token_idx, (h, c) = self.decoder(context, token_idx, (h, c))
            all_logits[:, t-1] = token_idx
            token_idx = token_idx.argmax(dim=1)
            teacher_mask = torch.rand(X.shape[0], device=y.device)<teacher_forcing_ratio
            token_idx = torch.where(teacher_mask, y[:, t], token_idx)
        return all_logits

    @torch.inference_mode()
    def greedy_decode(self, X, X_length, max_length, pad_idx):
        all_logits=[]
        token_idx = torch.full((X.shape[0], ), self.sos_idx, device=X.device)
        X, precomputed_X, (h, c), mask = self.encoder(X, X_length, pad_idx)
        check = torch.full((X.shape[0], ), False, dtype=torch.bool, device=X.device)
        for t in range(max_length):
            context=self.attention(X, precomputed_X, h, mask=mask)
            token_idx, (h, c) = self.decoder(context, token_idx, (h, c))
            token_idx = token_idx.argmax(dim=1)
            token_idx = token_idx.masked_fill(check, self.pad_idx)
            all_logits.append(token_idx.unsqueeze(0))
            check = check | (token_idx==self.eos_idx)
            if check.all():
                break
        return torch.permute(torch.concat(all_logits, dim=0), (1, 0))

    @torch.inference_mode()
    def beam_search(self, X, X_length, beam_width, max_length, pad_idx):
        B, K = X.shape[0], beam_width
        all_logits=[]
        y = torch.full((B, K, 1), self.sos_idx, dtype=torch.long, device=X.device)
        priority = torch.zeros((B,  K), device=X.device)
        priority[:, 1:] = -1e9
        y_length = torch.ones((B, K), device=X.device)
        X, precomputed_X, (h, c), mask = self.encoder(X, X_length, pad_idx)
        X = X.unsqueeze(1).repeat(1, K, 1, 1).view(B*K, X.shape[-2], X.shape[-1])
        if precomputed_X is not None:
            precomputed_X = precomputed_X.unsqueeze(1).repeat(1, K, 1, 1).view(B*K, precomputed_X.shape[-2], precomputed_X.shape[-1])
        h, c = h.repeat(1, K, 1).view(h.shape[0]*K, 1, -1), c.repeat(1, K, 1).view(h.shape[0]*K, 1, -1)
        mask = mask.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)
        check = torch.full((B, K), False, dtype=torch.bool, device=X.device)
        for t in range(max_length):
            context=self.attention(X, precomputed_X, h, mask=mask)
            token_idx, (h, c) = self.decoder(context, y[:, :, t].view(-1), (h, c))
            value = nn.functional.log_softmax(token_idx, dim=1).view(B, K, -1)
            value[check] = -1e9
            value[check, self.pad_idx] = 0
            tot_value = priority.unsqueeze(2)+value
            tot_value /= y_length.unsqueeze(2)
            tot_value = tot_value.view(B, -1)
            top_values, top_indices = torch.topk(tot_value, K, dim=1)
            beam_indices = top_indices//self.translated_vocabulary_size
            token_indices = top_indices%self.translated_vocabulary_size
            y = torch.gather(y, 1, beam_indices.unsqueeze(2).repeat(1, 1, t+1))
            y = torch.concat([y, token_indices.unsqueeze(-1)], dim=2)
            priority = top_values
            h, c = h.view(B, K, -1), c.view(B, K, -1)
            h, c = torch.gather(h, 1, beam_indices.unsqueeze(2).repeat(1, 1, h.shape[-1])), torch.gather(c, 1, beam_indices.unsqueeze(2).repeat(1, 1, c.shape[-1]))
            h, c = h.view(B*K, 1, -1), c.view(B*K, 1, -1)
            check = check | (y[:, :, -1]==self.eos_idx)
            y_length += ~check
            if check.all():
                break
        y = y[torch.arange(y.shape[0], device=y.device), (priority/y_length).argmax(dim=1), :]
        return y

if __name__ == "__main__":
    import sys
    Encoder = pretrained_encoder(192, True)
    print("sucess")
    #sys.exit()
    from vocabulary import Vocabulary, getEmbeddingMatrix
    from data import training_dataset, collate_fn
    from transformers import PreTrainedTokenizerFast
    tokenizer=PreTrainedTokenizerFast.from_pretrained("../.tokenizer/Hindi")
    #vocabulary = Vocabulary("../dataset/Hindi_train.csv")
    #M = getEmbeddingMatrix(vocabulary)
    train_dataset=training_dataset("../dataset/Hindi_train.csv", tokenizer, pretrained=True)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=lambda batch: collate_fn(batch, train_dataset.src_tokenizer.pad_token_id, tokenizer.pad_token_id, pretrained=True))
    #Encoder = encoder(embedding_dim=100, hidden_dim=96, embedding_matrix=M, pad_idx=vocabulary.pad_idx)
    Decoder = decoder(context_dim=768, hidden_dim=192, embedding_dim=192, vocabulary_size=8000)
    Attention = attention(encoder_dim=768, decoder_dim=192)
    model=seq2seq(Encoder, Decoder, Attention, 8000, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    for batch in train_loader:
        X, X_length, y, y_length = batch["X"], batch["X_length"], batch["y"], batch["y_length"]
        model(X, X_length, y, y_length, pad_idx=train_dataset.src_tokenizer.pad_token_id, teacher_forcing_ratio=0.6)
        with torch.no_grad():
            model.beam_search(X, X_length, beam_width=5, max_length=10, pad_idx=train_dataset.src_tokenizer.pad_token_id)
        break
    print("end of script")
