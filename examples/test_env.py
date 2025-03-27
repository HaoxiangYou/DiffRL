import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from pathlib import Path

import hydra
import numpy as np
import torch
import copy
from tqdm import tqdm
import time

import dm_env
from dm_env import specs
from tensorboardX import SummaryWriter
import envs
from externals.drqv2 import utils
from externals.drqv2 import dmc
from externals.drqv2.logger import Logger
from externals.drqv2.replay_buffer import ReplayBufferStorage, make_replay_loader
from externals.drqv2.video import TrainVideoRecorder, VideoRecorder
from utils.common import *

import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type = str, default = 'AntEnv')
parser.add_argument('--num-envs', type = int, default = 64)

args = parser.parse_args()

seeding()

env_fn = getattr(envs, args.env)

env = env_fn(num_envs = args.num_envs, \
            device = 'cuda:0', \
            seed = 0, \
            stochastic_init = True, \
            MM_caching_frequency = 16, \
            no_grad = True)

obs = env.reset()

num_actions = env.num_actions

t_start = time.time()

reward_episode = 0.
for i in range(1000):
    actions = torch.randn((args.num_envs, num_actions), device = 'cuda:0')
    obs, reward, done, info = env.step(actions)
    reward_episode += reward

t_end = time.time()

print('fps = ', 1000 * args.num_envs / (t_end - t_start))
print('mean reward = ', reward_episode.mean().detach().cpu().item())

print('Finish Successfully')

