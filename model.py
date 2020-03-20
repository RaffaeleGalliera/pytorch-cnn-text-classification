import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The in_channels argument is the number of "channels" in your image going into the convolutional layer.
        # In actual images this is usually 3 (one channel for each of the red, blue and green channels),
        # however when using text we only have a single channel, t
        # he text itself. The out_channels is the number of filters and the kernel_size is the size of the filters.
        # Each of our kernel_sizes is going to be [n x emb_dim] where $n$ is the size of the n-grams.

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        # In PyTorch, RNNs want the input with the batch dimension second, whereas CNNs want the batch dimension first
        # - we do not have to permute the data here as we have already set batch_first = True in our TEXT field.
        # We then pass the sentence through an embedding layer to get our embeddings.
        # The second dimension of the input into a nn. Conv2d layer must be the channel dimension.
        # As text technically does not have a channel dimension,
        # we unsqueeze our tensor to create one.
        # This matches with our in_channels=1 in the initialization of our convolutional layers.

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def predict_class(self, sentence, nlp, dataset, device, min_len=4):
        self.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [dataset.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        tensor = tensor.permute(1, 0)
        preds = self(tensor)
        max_preds = preds.argmax(dim=1)

        return max_preds.item()

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
