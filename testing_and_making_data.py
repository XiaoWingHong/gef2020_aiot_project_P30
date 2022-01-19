from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
from collections import deque
import cv2
import os
import random
import tensorflow as tf

BOX_MOVE = "model"  # random or model

MODEL_NAME = "models/model"  # model path

IsCollectData = False

if BOX_MOVE == "model":
    model = tf.keras.models.load_model(MODEL_NAME)
    reshape = (-1, 8, 40)
    model.predict(np.zeros((16,8,40)).reshape(reshape))

ACTION = 'right'

FFT_MAX_HZ = 40

HM_SECONDS = 15 #duration
TOTAL_ITERS = HM_SECONDS*25  # samples per second


last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

WIDTH = 800
HEIGHT = 800
SQ_SIZE = 50
MOVE_SPEED = 1

square = {'x1': int(int(WIDTH)/2-int(SQ_SIZE/2)),
          'x2': int(int(WIDTH)/2+int(SQ_SIZE/2)),
          'y1': int(int(HEIGHT)/2-int(SQ_SIZE/2)),
          'y2': int(int(HEIGHT)/2+int(SQ_SIZE/2))}


box = np.ones((square['y2']-square['y1'], square['x2']-square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

total = 0
correct = 0

channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(8): # each of the 8 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))

    env = np.zeros((WIDTH, HEIGHT, 3))

    env[:,HEIGHT//2-5:HEIGHT//2+5,:] = horizontal_line
    env[WIDTH//2-5:WIDTH//2+5,:,:] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2']] = box

    cv2.imshow('', env)
    cv2.waitKey(1)
    if BOX_MOVE == "model":
        network_input = np.array(channel_data).reshape(reshape)
        out = model.predict(network_input)
        print(out[0])

    if BOX_MOVE == "random":
        move = random.choice([-1,0,1])
        square['x1'] += move
        square['x2'] += move

    elif BOX_MOVE == "model":
        choice = np.argmax(out)
        if choice == 0:
            if ACTION == "left":
                correct += 1
            square['x1'] -= MOVE_SPEED
            square['x2'] -= MOVE_SPEED

        else:
            if ACTION == "right":
                correct += 1
            square['x1'] += MOVE_SPEED
            square['x2'] += MOVE_SPEED

    total += 1

    channel_datas.append(channel_data)

cv2.destroyAllWindows()

if IsCollectData:
    datadir = "data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    actiondir = f"{datadir}/{ACTION}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    print(len(channel_datas))

    print(f"saving {ACTION} data...")
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
    print("done.")
print("accuracy: " + str(correct/total*100))
