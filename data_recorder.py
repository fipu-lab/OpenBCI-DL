from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import os


OBJECT = input("Upisi direktorij osoba/akcija: ")

ITER_PER_SEC = 90 # iteracije po sekundi
HM_SECONDS = 60  # Broj sekunda
TOTAL_ITERS = HM_SECONDS*ITER_PER_SEC

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("...stream resolved, starting!")

print("starting focus time...")
for _ in range(ITER_PER_SEC * 1): #focus time 3 sec
    for _ in range(16):
        inlet.pull_sample()
print("...focus time over")

print(f"started data gatther for object {OBJECT}")
channel_datas = []
for i in range(TOTAL_ITERS):
    channel_data = []
    for j in range(16):
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:60]) # do 60hz

    channel_datas.append(channel_data)

datadir = "data"
if not os.path.exists(datadir):
    os.mkdir(datadir)

objectdir = f"{datadir}/{OBJECT}"
if not os.path.exists(objectdir):
    os.makedirs(objectdir, exist_ok=True)

print(len(channel_datas))
print(f"Saving data for object {OBJECT}")
np.save(os.path.join(objectdir, f"{int(time.time())}.npy"), np.array(channel_datas))
print(f"{OBJECT} done")





