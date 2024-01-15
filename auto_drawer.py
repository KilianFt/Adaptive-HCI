
import time
import math
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from tempfile import TemporaryDirectory
from sklearn.model_selection import train_test_split

from adaptive_hci.datasets import get_omniglot_moves
from adaptive_hci.controllers import TransformerModel


def batchify(data: Tensor, bsz: int, device: str) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def train(model: nn.Module, train_data, criterion, optimizer, scheduler, bptt, ntokens, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor, criterion, ntokens, bptt) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def main():
    canvas_size = 30
    omniglot_dir = Path('./datasets/omniglot')
    encoded_moves_data = get_omniglot_moves(omniglot_dir, canvas_size=canvas_size, max_initial_value=120)

    move_map = [[-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
                [0, 0]]
    
    # TODO show result in end
    # plot_encoded_moves(encoded_moves_data[6], move_map, canvas_size=30)

    encoded_data_with_stop = [torch.cat((x, torch.tensor([4]))) for x in encoded_moves_data]

    train_list, val_list = train_test_split(encoded_data_with_stop)

    flat_train_data = torch.concat(train_list).type(torch.int)
    flat_val_data = torch.concat(val_list).type(torch.int)

    ntokens = 5
    device = 'mps'
    lr = 5.0  # learning rate
    batch_size = 20
    eval_batch_size = 10
    bptt = 35

    train_data = batchify(flat_train_data, batch_size, device=device)  # shape ``[seq_len, batch_size]``
    val_data = batchify(flat_val_data, eval_batch_size, device=device)

    model = TransformerModel(ntoken=ntokens, d_model=4, nhead=2, d_hid=128,
                            nlayers=1, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 30

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model, train_data, criterion, optimizer, scheduler,
                  bptt, ntokens, epoch)
            val_loss = evaluate(model, val_data, criterion, ntokens, bptt)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
        model.load_state_dict(torch.load(best_model_params_path)) # load best model states

    torch.save(model, 'models/drawer_test.pt')


if __name__ == '__main__':
    main()
