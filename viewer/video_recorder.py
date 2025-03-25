import imageio
import os
import queue
import threading 

class VideoRecorder:
    def __init__(self, root_dir=None, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.frames = []
        # Queue for sequential processing
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Processes video saving jobs one by one."""
        while True:
            file_name, frames = self.queue.get()
            if file_name is None:
                break  # Exit the worker thread
            self._save_video(file_name, frames)
            self.queue.task_done()

    def _save_video(self, file_name, frames):
        """Saves the video file."""
        path = os.path.join(self.save_dir, file_name)
        imageio.mimsave(path, frames, fps=self.fps)

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
        """Queues the video saving job (one at a time)."""
        self.queue.put((file_name, self.frames[:]))  # Copy frames
        self.reset()  # Clear frames for the next recording

    def stop(self):
        """Stops the worker thread safely."""
        self.queue.put((None, None))  # Signal thread to exit
        self.worker_thread.join()