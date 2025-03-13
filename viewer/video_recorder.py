import imageio
import os 

class VideoRecorder:
    def __init__(self, root_dir=None, fps=20, height=256, width=256, camera_id=0):
        if root_dir is not None:
            self.save_dir = root_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.render_kwargs = dict(height=height, width=width, camera_id=camera_id)
        self.frames = []

    def update_save_dir(self, root_dir):
        if root_dir is not None:
            self.save_dir = root_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

    def append(self, frame):
        self.frames.append(frame)

    def reset(self):
        self.frames = []

    def save(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        imageio.mimsave(path, self.frames, fps=self.fps)