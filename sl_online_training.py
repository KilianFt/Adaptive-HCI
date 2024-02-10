import time
import collections
import pickle
from datetime import datetime
from collections import deque
import multiprocessing

import pygame
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from adaptive_hci.datasets import to_tensor_class_dataset
from adaptive_hci.controllers import SingleLabelPlModel
from realtime_pred_test import worker
from interface.game import load_emg_decoder
from configs import BaseConfig

emg_queue = multiprocessing.Queue()

def predict(emg_decoder, emg_tensor):
    emg_pred_raw = emg_decoder(emg_tensor)
    emg_pred = F.softmax(emg_pred_raw, dim=-1)
    pred_idx = emg_pred.argmax(dim=-1)
    return pred_idx


def train_model(emg_decoder, emg, labels):
    print('Training model')
    train_offline_adaption_dataset = to_tensor_class_dataset(emg, labels)
    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, batch_size=32,
                                  num_workers=6)

    pl_model = SingleLabelPlModel(
        emg_decoder,
        n_labels=5,
        lr=1e-4,
        n_frozen_layers=0,
        threshold=0.5,
        metric_prefix='online_sl/',
        criterion_key='ce',
    )
    trainer = L.Trainer(max_epochs=10, log_every_n_steps=1,
                         enable_checkpointing=False, accelerator='mps')

    trainer.fit(model=pl_model, train_dataloaders=train_dataloader)


def main():
    pygame.init()

    myo_process = multiprocessing.Process(target=worker, args=(emg_queue,))
    myo_process.start()

    WIDTH, HEIGHT = 640, 480
    WHITE = (255, 255, 255)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    config = BaseConfig()
    emg_decoder_state_dict_file = './models/general_emg_vit_state_dict.pt'
    emg_decoder = load_emg_decoder(emg_decoder_state_dict_file, config)

    buffer = deque(maxlen=config.window_size)

    stride = config.window_size - config.overlap
    while len(buffer) < config.window_size:
        while not emg_queue.empty():
            emg = list(emg_queue.get())
            buffer.append(emg)

    n_new_samples = stride

    directions = ['rest', 'up', 'left', 'down', 'right', '']
    train_sequence = [0, 1, 0, 2, 0, 3, 0, 4, 0]
    current_sequence_idx = 0
    timeout = 1

    dataset = collections.defaultdict(list)

    emg_pred = 0
    last_window_time = time.time()
    last_change_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        if current_sequence_idx % (len(train_sequence) - 1) == 0 and current_sequence_idx > 0:
            labels = torch.tensor(dataset['labels'], dtype=torch.long)
            emg = torch.tensor(dataset['emg'], dtype=torch.float32) / 1500
            print(emg.shape)
            print(labels.shape)
            predictions = predict(emg_decoder, emg)

            print(predictions.shape)
            accuracy = (predictions == labels).type(torch.float32).mean()
            print('accuracy', accuracy)

            train_model(emg_decoder, emg, labels)
            current_sequence_idx = 0

            # TODO don't delete dataset
            dataset = collections.defaultdict(list)


        if time.time() - last_change_time >= timeout:
            current_sequence_idx = (current_sequence_idx + 1) % len(train_sequence)
            last_change_time = time.time()

        screen.fill(WHITE)

        current_label_idx = train_sequence[current_sequence_idx]

        if not emg_queue.empty():
            emg = list(emg_queue.get())
            buffer.append(emg)
            n_new_samples += 1

        if n_new_samples >= stride:
            win_dt = time.time() - last_window_time
            print(f'new window {win_dt:.2f}')
            last_window_time = time.time()

        #     emg_pred = predict(emg_decoder, buffer)
        #     n_new_samples = 0

            dataset['emg'].append(buffer)
            dataset['labels'].append(current_label_idx)
            # dataset['prediction'].append(emg_pred)

        # matches = sum([1 for x, y in zip(dataset['label'][-20:], dataset['prediction'][-20:]) if x == y])
        # accuracy = matches / len(dataset['label'])
        # accuracy = 0

        # display text
        # direction_text = font.render(f"Accuracy: {accuracy:.2f}", False, (0,0,0))
        # text_rect = direction_text.get_rect()
        # text_rect.center = (WIDTH // 2, (HEIGHT // 6))
        # screen.blit(direction_text, text_rect)

        direction_text = font.render(directions[current_label_idx], False, (255,0,0))
        text_rect = direction_text.get_rect()
        text_rect.center = (WIDTH // 2, HEIGHT // 4)
        screen.blit(direction_text, text_rect)

        next_sequence_idx = (current_sequence_idx + 1) % len(train_sequence)
        next_label_idx = train_sequence[next_sequence_idx]
        direction_text = font.render(f"next: {directions[next_label_idx]}", False, (150,50,50))
        text_rect = direction_text.get_rect()
        text_rect.center = (WIDTH // 2, 100 + (HEIGHT // 4))
        screen.blit(direction_text, text_rect)

        pred_direction_text = font.render(directions[emg_pred], False, (0,0,0))
        text_rect = pred_direction_text.get_rect()
        text_rect.center = (WIDTH // 2, (HEIGHT * 3) // 4)
        screen.blit(pred_direction_text, text_rect)

        # Show the time since last_change_time
        time_since_change = time.time() - last_change_time
        time_till_change = timeout - time_since_change
        time_text = font.render(f'Time since last change: {time_till_change:.1f}s', False, (0,0,0))
        time_rect = time_text.get_rect()
        time_rect.center = (WIDTH // 2, (HEIGHT * 5) // 6)
        screen.blit(time_text, time_rect)
        pygame.display.flip()

        clock.tick(60)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    out_file = f'datasets/interactive_sl/{dt_string}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(dataset, f)

    pygame.quit()


if __name__ == '__main__':
    main()