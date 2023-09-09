# real-time prediction with torch model

'''
Can plot EMG data in 2 different ways
change DRAW_LINES to try each.
Press Ctrl + C in the terminal to exit 
'''

import numpy as np
import multiprocessing

import torch
from vit_pytorch import ViT

from pyomyo import Myo, emg_mode

device = 'mps'

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

    model = torch.load('models/pretained_vit.pt').to(device)

    move_map = {
        0: 'Neutral', 
        1: 'Radial Deviation', 
        2: 'Wrist Flexion', 
        3: 'Ulnar Deviation', 
        4: 'Wrist Extension',
        }

    emg_min = -128
    emg_max = 127

    emg_buffer = []

    try:
        while True:
            while not(q.empty()):
                emg = list(q.get())
                norm_emg = np.interp(emg, (emg_min, emg_max), (-1, +1))
                # print(norm_emg)
                emg_buffer.append(norm_emg)

                if len(emg_buffer) > 200:
                    emg_window = np.array(emg_buffer[-200:], dtype=np.float32)

                    emg_window_tensor = torch.from_numpy(emg_window).unsqueeze(0).unsqueeze(0).to(device)
                    emg_window_tensor.swapaxes_(2, 3)
                    # print(emg_window_tensor.shape)
                    predictions = model(emg_window_tensor)
                    pred_idx = torch.argmax(predictions).cpu().item()
                    print(move_map[pred_idx])
                    # print(torch.argmax(predictions))

    except KeyboardInterrupt:
        print("Quitting")
        quit()