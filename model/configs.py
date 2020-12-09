import numpy as np

# config data
ANCHORS = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]
NUM_CLASS = 5

# config model
INPUT_SIZE = 448

# config training
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 0.001

MAX_BOX_PER_IMAGE = 32
IOU_LOSS_THRESH = 0.5
IOU_SCALE_THRESH = 0.3
# large GRID_SIZE is used small obj and opposite
GRID_SIZE = [7, 14, 28]  # = output of HEAD ex: GRID_SIZE = 16 (GRID_SIZE = INPUT_SIZE/GRID_STRIDE)
ANCHORS_PER_OBJ = 3
