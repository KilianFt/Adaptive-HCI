# real-time prediction with torch model

'''
Can plot EMG data in 2 different ways
change DRAW_LINES to try each.
Press Ctrl + C in the terminal to exit 
'''
import time
from collections import deque

import numpy as np
import multiprocessing
import torch
from vit_pytorch import ViT
from pyomyo import Myo, emg_mode

from controllers import SLOnlyController

device = 'mps'
emg_min = -128
emg_max = 127


# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q):
	m = Myo(mode=emg_mode.RAW)
	m.connect()
	
	def add_to_queue(emg, movement):
		q.put(emg)

	m.add_emg_handler(add_to_queue)
	
	def print_battery(bat):
		print("Battery level:", bat)

	m.add_battery_handler(print_battery)

	 # Orange logo and bar LEDs
	m.set_leds([128, 0, 0], [128, 0, 0])
	# Vibrate to know we connected okay
	m.vibrate(1)
	
	"""worker function"""
	while True:
		m.run()
	print("Worker Stopped")
    


# -------- Main Program Loop -----------
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    controller = SLOnlyController('models/pretrained_vit_onehot_mad.pt', device=device)

    move_map = {
        0: 'Neutral', 
        1: 'Radial Deviation', # up
        2: 'Wrist Flexion', # left
        3: 'Ulnar Deviation', # down
        4: 'Wrist Extension', # right
        }

    emg_buffer = deque(maxlen=200)
    # initialize as -150 cause first window needs 200 samples
    n_new_samples = -150
    last_time_window = time.time()

    try:
        while True:
            while n_new_samples < 50:
                while not(q.empty()):
                    emg = list(q.get())
                    norm_emg = np.interp(emg, (emg_min, emg_max), (-1, +1))
                    emg_buffer.append(norm_emg)
                    n_new_samples += 1
            
            # build new window every 50 new samples
            emg_window = np.array(emg_buffer, dtype=np.float32)
            emg_window = torch.tensor(emg_window).unsqueeze(0)
            outputs = controller.deterministic_forward(emg_window)

            predictions = outputs.cpu().detach().squeeze().numpy()
            predicted_labels = np.zeros_like(predictions)
            predicted_labels[predictions > 0.5] = 1

            pred_inds = np.argwhere(predicted_labels)
            for pred_idx in pred_inds:
                print(move_map[pred_idx[0]])

            current_time = time.time()
            dt_windows = current_time - last_time_window
            print('window dt', dt_windows)
            last_time_window = current_time

            n_new_samples = 0

    except KeyboardInterrupt:
        print("Quitting")
        p.terminate()
        p.join()
        quit()
