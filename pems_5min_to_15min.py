import numpy as np

pems = np.load('data/pems_speed_occupancy_5min.npz')
speed_5 = pems['speed']
occupancy_5 = pems['occupancy']
h_5, v = speed_5.shape
h_15 = int(h_5 / 3)
occupancy_15 = np.zeros([h_15, v], dtype=np.float32)
speed_15 = np.zeros([h_15, v], dtype=np.float32)
for i in range(h_15):
    occupancy_15[i] = np.sum(occupancy_5[i * 3:i * 3 + 2], 0)
    speed_15[i] = np.sum(speed_5[i * 3:i * 3 + 2], 0)

np.savez('data/pems_speed_occupancy_15min.npz', speed=speed_15, occupancy=occupancy_15)
