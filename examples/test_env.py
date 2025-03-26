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
from viewer.dmc_viewer import DMCViewer
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
import time
import yaml
from collections import defaultdict
from train_drqv2 import MakeDMfromShac as MakeDMfromShacSingleEnv
from train_drqv2_parallel import MakeDMfromShac as MakeDMfromShacMultiEnv

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.num_envs = cfg["params"]["config"]["num_actors"]
        self.img_height = cfg["params"]["config"].get("img_height", 84)
        self.img_width = cfg["params"]["config"].get("img_width", 84)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._num_episode_finished = 0
        self.time_report = TimeReport()
        self.writer = SummaryWriter(os.path.join(self.work_dir, 'tb'))
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.vis_obs_buffer = torch.zeros(
            (self.num_envs, 9, self.img_height , self.img_width ), device=self.device, dtype=torch.uint8, requires_grad=False)
        self.iter_count = 0
        self.step_count = 0
        self._current_episodes = [copy.deepcopy(defaultdict(list)) for _ in range(self.num_envs)]
        self.episode_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.global_step = 0

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def setup(self):
        # create logger
        # self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        # env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        train_env = MakeDMfromShacMultiEnv(self.cfg, False)
        eval_env = MakeDMfromShacMultiEnv(self.cfg, True)
        test_env = MakeDMfromShacSingleEnv(self.cfg)

        self.train_env = dmc.make_from_shac(train_env, self.cfg)
        self.eval_env = dmc.make_from_shac(eval_env, self.cfg)
        self.test_env = dmc.make_from_shac_single_thread(test_env, self.cfg)
        # create replay buffer
        action_spec = specs.BoundedArray((self.train_env.num_actions, ),
                                                   minimum=-1,
                                                   maximum=1,
                                                   dtype='float32',
                                                   name='action')
        self.data_specs = (self.train_env.observation_spec(),
                      action_spec, # self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(self.data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)
    
    def process_time_steps(self, store_time_steps):
        done_ids = []
        for idx, time_step in enumerate(store_time_steps):
            # store the observation 
            for spec in self.data_specs:
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episodes[idx][spec.name].append(value)
            if time_step.last():
                done_ids.append(idx)
                # if the current episode ends, we store the episode 
                episode = dict()
                for spec in self.data_specs:
                    value = self._current_episodes[idx][spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
                    # we only save episodes that finished
                self._current_episodes[idx] = copy.deepcopy(defaultdict(list))
                self.replay_storage._store_episode(episode)
        
        # Finally, process the reward for logging
        with torch.no_grad():
            self.episode_loss -= torch.tensor(self.train_env.raw_rew, dtype=torch.float32, device=self.device)
            if len(done_ids)>0:
                self.episode_loss_meter.update(self.episode_loss[done_ids])
                for done_env_id in done_ids:
                    if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                        print('ep loss error')
                        raise ValueError
                    self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                    self.episode_loss[done_env_id] = 0.
                
                self.train_env.reset(np.array(done_ids, dtype=np.int32))
        
        self._global_episode += len(done_ids)
        self._num_episode_finished += len(done_ids)

    def train(self):
        time_step_test = self.test_env.reset()
        time_step_train = self.train_env.reset()
        time_step_eval = self.eval_env.reset()
        test_list = []
        train_list = []
        eval_list = []
        import pdb
        for i in tqdm(range(5000)):
            with torch.no_grad(), utils.eval_mode(self.agent):
                self.vis_obs_buffer[:] = torch.tensor([time_step.observation for time_step in time_step_eval])
                action = self.agent.act(self.vis_obs_buffer,
                                        self.global_step,
                                        eval_mode=False)

            # take env step       
            time_step_train = self.train_env.step(action)
            time_step_eval = self.eval_env.step(action[0])
            time_step_test = self.test_env.step(action[0])
            

            # if not np.array_equal(time_step_test.observation, time_step_train[0].observation):
            #     print('test and train envs are not equal')
            #     pdb.set_trace()
            # elif not (time_step_test.last() == time_step_train[0].last()):
            #     print('test and train envs last are not equal')
            #     pdb.set_trace()
            # elif np.abs(time_step_test.reward - time_step_train[0].reward) > 0.1:
            #     print(f'test {time_step_test.reward} and train {time_step_train[0].reward} reward are not equal')
            #     pdb.set_trace()
            # elif not np.array_equal(time_step_test.observation, time_step_eval[0].observation):
            #     print('test and eval envs are not equal')
            #     pdb.set_trace()
            # elif not (time_step_test.last() == time_step_eval[0].last()):
            #     print('test and eval envs last are not equal')
            #     pdb.set_trace() 
            # elif  np.abs(time_step_test.reward - time_step_eval[0].reward) > 0.1:
            #     print(f'test {time_step_test.reward} and train {time_step_eval[0].reward} reward are not equal')
            #     pdb.set_trace()

            self.process_time_steps(time_step_train)

            if self.train_env.done_envs.shape[0] > 0:
                self.train_env.reset(self.train_env.done_envs)
            if self.eval_env.done_envs.shape[0] > 0:
                self.eval_env.reset(self.eval_env.done_envs)
            
            if i % 200 == 0:
                pdb.set_trace()
                # next(self.replay_iter)

        import pdb; pdb.set_trace()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def close(self):
        self.writer.close()


@hydra.main(config_path='cfg/drqv2', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()