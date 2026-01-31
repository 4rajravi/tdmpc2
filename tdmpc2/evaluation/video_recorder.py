import imageio
import os

import warnings
warnings.filterwarnings("ignore")


class VideoRecorder:

    def __init__(self, save_dir, fps=30):
        self.save_dir = save_dir
        self.frames = []
        self.fps = fps
        os.makedirs(save_dir, exist_ok=True)

    def record(self, frame):
        self.frames.append(frame)

    def save(self, filename):
        path = os.path.join(self.save_dir, filename)
        imageio.mimsave(path, self.frames, fps=self.fps)
        self.frames = []
