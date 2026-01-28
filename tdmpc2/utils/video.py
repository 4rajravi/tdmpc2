import imageio
import os


class VideoRecorder:
    def __init__(self, dir_path, enabled=True):
        self.enabled = enabled
        self.dir = dir_path
        self.frames = []

        if enabled:
            os.makedirs(dir_path, exist_ok=True)

    def record(self, frame):
        if self.enabled:
            self.frames.append(frame)

    def save(self, filename):
        if not self.enabled or len(self.frames) == 0:
            return

        path = os.path.join(self.dir, filename)
        imageio.mimsave(path, self.frames, fps=30)
        self.frames = []
