from skimage.color import rgb2gray
from collections import deque
import numpy as np


FRAMES_STACK_SIZE = 4


def preprocess_frame(frame):
    """ Frame preprocessing - grayscaling, resizing, normalization """
    gray = rgb2gray(frame)
    cropped = gray[8:-12, 4:-12]
    normalized = cropped / 255.0
    return normalized.transpose((1, 0))


def stack_frames(raw_frame, stacked_frames=None):
    frame = preprocess_frame(raw_frame)

    if not stacked_frames:
        stacked_frames = deque([frame for _ in range(FRAMES_STACK_SIZE)],
                               maxlen=FRAMES_STACK_SIZE)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames
