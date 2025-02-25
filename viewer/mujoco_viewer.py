# Image rendering in dm_control styles
import numpy as np
import sys, os
import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from utils.conventions_utils import shac_pos_to_mujoco, shac_quat_to_mujoco
from dm_control.mujoco.engine import Physics


class MujocoViewer:
    def __init__(self, file_path:str, height=84, width=84, camera_id=0, root_pos_index=None, root_quat_index=None):
        self.physics = Physics.from_xml_path(file_path)
        self.render_kwargs = dict(height=height, width=width, camera_id=camera_id)
        self.root_pos_index = root_pos_index
        self.root_quat_index = root_quat_index

    def render(self, qpos:np.ndarray, render_kwargs=None):
        qpos = self.convention_transform(qpos)
        np.copyto(self.physics.data.qpos, qpos)
        self.physics.forward()
        if render_kwargs is None:
            pixels = self.physics.render(**self.render_kwargs)
        else:
            pixels = self.physics.render(**render_kwargs)
        return pixels

    def convention_transform(self, qpos:np.ndarray):
        if self.root_pos_index is not None:
            qpos[self.root_pos_index:self.root_pos_index+3] = shac_pos_to_mujoco(qpos[self.root_pos_index:self.root_pos_index+3])
        if self.root_quat_index is not None:
            qpos[self.root_quat_index:self.root_quat_index+4] = shac_quat_to_mujoco(qpos[self.root_quat_index:self.root_quat_index+4])
        return qpos

if __name__ == "__main__":
    import pickle
    import imageio

    xml_file = os.path.join(project_dir, "envs/assets/ant.xml")
    traj_path = os.path.join(project_dir, "examples/logs/Ant/dac/0/traj_final_policy.pkl")
    output_path = os.path.join(project_dir, "examples/logs/Ant/dac/0/final_policy.mp4")

    viewer = MujocoViewer(xml_file, camera_id=2, root_pos_index=0, root_quat_index=3)
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
        dt = traj["dt"]
        joint_qs = traj["joint_q"][:-1]

    frames = []
    for q in joint_qs[:,1,:]:
        frames.append(viewer.render(q, dict(height=256, width=256, camera_id=2)))

    imageio.mimsave(output_path, frames, fps=int(1/dt))
