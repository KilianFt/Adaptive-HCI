from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.functional import F

import lightning.pytorch as pl
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT

from adaptive_hci.utils import get_accelerator


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)


def get_criterion(criterion_key):
    print(f"Using {criterion_key} loss")
    # how do we deal with keys? can we use enum in sweep config?
    if criterion_key == 'mse':
        return torch.nn.MSELoss()
    elif criterion_key == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"{criterion_key} loss not supported")


def multilabel_at_least_one_match(y_true, y_pred):
    ''' Is correct if at least one label is predicted'''
    intersection = (y_true * y_pred).sum(dim=-1)
    at_least_one_correct = (intersection > 0).float()
    return at_least_one_correct.mean()


class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr, n_frozen_layers: int, threshold: float, metric_prefix: str = '',
                 criterion_key: str = 'bce'):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.threshold = threshold
        self.metric_prefix = metric_prefix
        self.criterion = get_criterion(criterion_key)
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.accuracy_metric = Accuracy(task='binary')
        self.step_count = 0
        if n_frozen_layers is not None:
            self.freeze_layers(n_frozen_layers)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        # reset grad
        # this is fine for ViT as by default all params require grad
        for param in self.model.parameters():
            param.requires_grad = True

        # freeze desired ones
        if n_frozen_layers >= 1:
            for param in self.model.to_patch_embedding.parameters():
                param.requires_grad = False
            for param in self.model.dropout.parameters():
                param.requires_grad = False

        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), len(self.model.transformer.layers))):
                for param in self.model.transformer.layers[layer_idx].parameters():
                    param.requires_grad = False

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log(f"{self.metric_prefix}train/loss", loss)
        self.log(f"{self.metric_prefix}train/step", self.step_count)
        self.step_count += 1
        return loss

    def get_per_label_accuracies(self, outputs, targets):
        num_targets = targets.shape[1]

        binary_outputs = (outputs >= self.threshold).int()
        binary_targets = (targets >= self.threshold).int()

        per_labels_accuracies = []

        for label_idx in range(num_targets):
            label_acc = self.accuracy_metric(binary_outputs[:, label_idx], binary_targets[:, label_idx])
            per_labels_accuracies.append(label_acc)

        return torch.tensor(per_labels_accuracies)

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        one_match = multilabel_at_least_one_match(y_true=targets, y_pred=outputs)
        val_loss = F.mse_loss(outputs, targets)
        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/f1', val_f1, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/one_match', one_match, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/step', self.step_count, prog_bar=True)

        per_label_accuracies = self.get_per_label_accuracies(outputs, targets)
        for label_idx, per_label_acc in enumerate(per_label_accuracies):
            self.log(f'{self.metric_prefix}validation/acc_label_{label_idx}', per_label_acc, prog_bar=True)

        return val_loss, val_acc, val_f1, per_label_accuracies

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# drawing model
# inspired from https://github.com/wingedsheep/transformer/blob/main/main.py#L496

class TokenEmbedding(torch.nn.Module):
    """
    PyTorch module that converts tokens into embeddings.

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, number_of_tokens):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=number_of_tokens,
            embedding_dim=d_model
        )

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(torch.nn.Module):
    """
    Pytorch module that creates a positional encoding matrix. This matrix will later be added to the
    transformer's input embeddings to provide a sense of position of the sequence elements.
    """

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Creates a positional encoding matrix of size (max_sequence_length, d_model).
        """

        # Initialize positional encoding matrix
        positional_encoding = np.zeros((self.max_sequence_length, self.d_model))

        # Calculate positional encoding for each position and each dimension
        for pos in range(self.max_sequence_length):
            for i in range(0, self.d_model, 2):
                # Apply sin to even indices in the array; indices in Python start at 0 so i is even.
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))

                if i + 1 < self.d_model:
                    # Apply cos to odd indices in the array; we add 1 to i because indices in Python start at 0.
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.d_model)))

        # Convert numpy array to PyTorch tensor and return it
        return torch.from_numpy(positional_encoding).float().to(get_accelerator())

    def forward(self, x):
        """
        Adds the positional encoding to the input embeddings at the corresponding positions.
        """
        # Add positional encodings to input embeddings. The ":" indexing ensures we only add positional encodings up
        # to the length of the sequence in the batch. x.size(0) is the batch size, so this is a way to make sure
        # we're not adding extra positional encodings.
        return x + self.positional_encoding[:x.size(1), :]


class MaskedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, head_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embedding_dimension)
        output dimension is: (batch_size, sequence_length, head_dimension)
        mask dimension is: (batch_size, sequence_length)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        # Scale the attention weights.
        attention_weights = attention_weights / np.sqrt(self.head_dimension)

        # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
        # This will make the softmax output 0 for these values.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dimension)
        return torch.bmm(attention_scores, value)


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)


class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))


class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_dimension)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # Layer normalization 1
        normalized_x = self.layer_normalization_1(x)

        # Multi headed self attention
        attention_output = self.multi_headed_self_attention(normalized_x, mask)

        # Residual output
        residual_output = x + attention_output

        # Layer normalization 2
        normalized_residual_output = self.layer_normalization_2(residual_output)

        # Feed forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        # Dropout
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)

        # Residual output
        return residual_output + feed_forward_output


class DecoderStack(torch.nn.Module):
    """
    The decoder stack consists of multiple decoder layers in sequence.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)])

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs


class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x):
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output


class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """

    def __init__(
            self,
            number_of_tokens,  # The number of tokens in the vocabulary
            max_sequence_length=512,  # The maximum sequence length to use for attention
            embedding_dimension=512,  # The dimension of the token embeddings
            number_of_layers=6,  # The number of decoder layers to use
            number_of_heads=4,  # The number of attention heads to use
            feed_forward_dimension=None,  # The dimension of the feed forward layer
            dropout_rate=0.1,  # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if feed_forward_dimension is None:
            # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)

        # Create the decoder stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

    def forward(self, x, mask):
        # Compute the token embeddings
        # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)

        # Compute the positional encoding
        # positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)

        # Post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)

        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'number_of_tokens': self.number_of_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'number_of_layers': self.number_of_layers,
            'number_of_heads': self.number_of_heads,
            'feed_forward_dimension': self.feed_forward_dimension,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, path)

    @staticmethod
    def load_checkpoint(path) -> 'LanguageModel':
        checkpoint = torch.load(path)
        model = LanguageModel(
            number_of_tokens=checkpoint['number_of_tokens'],
            max_sequence_length=checkpoint['max_sequence_length'],
            embedding_dimension=checkpoint['embedding_dimension'],
            number_of_layers=checkpoint['number_of_layers'],
            number_of_heads=checkpoint['number_of_heads'],
            feed_forward_dimension=checkpoint['feed_forward_dimension'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(get_accelerator())


class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.
    """

    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = self.model.max_sequence_length

    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inp, mask)
        return output, target

    def next_token_probabilities(self, x, mask, temperature=1.0):
        """
        Calculate the token probabilities for the next token in the sequence.
        """
        logits = self.model(x, mask)[:, -1]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply the softmax
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def save_checkpoint(self, path):
        self.model.save_checkpoint(path)

    @staticmethod
    def load_checkpoint(path) -> 'AutoregressiveWrapper':
        model = LanguageModel.load_checkpoint(path)
        return AutoregressiveWrapper(model).to(get_accelerator())


class LightningAutoregressor(pl.LightningModule):
    def __init__(self, model, lr, n_tokens, metric_prefix=''):
        super(LightningAutoregressor, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.metric_prefix = metric_prefix
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=n_tokens)
        self.step_count = 0

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        sample, mask = batch
        out, target = self.model(sample, mask)
        loss = self.criterion(out.transpose(1,2), target)
        self.log(f"{self.metric_prefix}train/loss", loss)
        self.log(f"{self.metric_prefix}train/step", self.step_count)
        self.step_count += 1
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        sample, mask = batch
        out, target = self.model(sample, mask)
        val_loss = self.criterion(out.transpose(1,2), target)
        val_acc = self.accuracy_metric(out.transpose(1,2), target)
        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/step', self.step_count, prog_bar=True)
        return val_loss
    
    # TODO compare SGD and Adam
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)