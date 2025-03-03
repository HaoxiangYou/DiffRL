# Image rendering in dm_control styles
import numpy as np
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from dm_control.mujoco.engine import Physics

class DMCViewer:
    def __init__(self, file_path:str, height=84, width=84, camera_id=0, root_pos_index=None, root_quat_index=None):
        self.physics = Physics.from_xml_path(file_path)
        self.render_kwargs = dict(height=height, width=width, camera_id=camera_id)
        self.root_pos_index = root_pos_index
        self.root_quat_index = root_quat_index

    def render(self, qpos:np.ndarray, render_kwargs=None):
        """
        qpos: joint qpos in mujoco conventions
        """
        np.copyto(self.physics.data.qpos, qpos)
        self.physics.forward()
        if render_kwargs is None:
            pixels = self.physics.render(**self.render_kwargs)
        else:
            pixels = self.physics.render(**render_kwargs)
        return pixels