import numpy as np
import mujoco

def shac_pos_to_mujoco(pos: np.ndarray) -> np.ndarray:
    """
    Transform shac position to mujoco position
    shac zxy corresponding to mujoco xyz
    """
    transformed_pos = np.empty(3)
    mujoco.mju_rotVecQuat(transformed_pos, pos, np.array([0.5,0.5,0.5,0.5]))
    return transformed_pos

def shac_quat_to_mujoco(quat:np.ndarray) -> np.ndarray:
    """
    Transform shac quat into mujoco quat:
    shac do [xyzw]
    mujoco in [wxyz]
    additionally apply change of coordinates:
    shac zxy corresponding to mujoco xyz
    """
    transformed_quat = np.empty(4)
    mujoco.mju_mulQuat(transformed_quat, np.array([0.5,0.5,0.5,0.5]), quat[[3,0,1,2]])
    return transformed_quat
