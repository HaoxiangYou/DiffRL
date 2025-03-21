# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class MakeDMfromShac(dm_env.Environment):
    def __init__(self, cfg, eval):
        self.eval = eval
        self.cfg = cfg
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        seeding(cfg["params"]["general"]["seed"])
        self.env =  env_fn(num_envs = 1 if self.eval else cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            render = cfg["params"]["general"]["render"], \
                            vis_obs = cfg["params"]["config"].get("vis_obs", False), \
                            img_height = cfg["params"]["config"].get("img_height", 84),\
                            img_width = cfg["params"]["config"].get("img_width", 84),\
                            render_mode = cfg["params"]["config"]["player"].get("render_mode", 'usd') , \
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            no_grad = True)
        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_state_obs = ', self.env.num_state_obs)
        print('num_vis_obs =', self.env.num_vis_obs)
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions
        self.render_size = 256 # fixed due to the data mismatch with TrainVideoRecorder
        self.camera_id = 0 # render camera id. 
        self.render_kwargs = dict(height=self.render_size, width=self.render_size, camera_id=self.camera_id)
        self.dmc_render_model = cfg["params"]["config"]["dmc_render_model"]
        self.dmc_render = DMCViewer(file_path=os.path.join(project_dir, f"envs/assets/{self.dmc_render_model}.xml"), 
                                            camera_id=0, height=self.render_size, width=self.render_size)
        self.device = cfg["params"]["general"]["device"]
        self.raw_rew = np.zeros((self.env.num_envs)) 
        if hasattr(self.env, 'observation_spec'):
            self._observation_spec = self._env.observation_spec()
        else:
            self._observation_spec = specs.BoundedArray(self.env.num_vis_obs,
                                                        minimum= 0,
                                                        maximum= 255,
                                                        dtype='uint8',
                                                        name='observation')
        if hasattr(self.env, 'action_spec'):
            self._action_spec = self._env.action_spec()
        else:
            self._action_spec = specs.BoundedArray((self.env.num_actions, ),
                                                   minimum=-1,
                                                   maximum=1,
                                                   dtype='float32',
                                                   name='action')
        self._reward_spec = specs.Array(shape=(), dtype=np.dtype('float32'), name='reward')
        self._discount_spec = specs.BoundedArray(
        shape=(), dtype='float32', minimum=0.0, maximum=1.0, name='discount')
        if hasattr(self.env, 'discount_spec'):
            self._discount_spec = self.env.discount_spec()

    def reset(self, env_ids = None, force_reset = True):
        # return stacked observation (9 * width * height)
        self.env.clear_grad()
        obs = self.env.reset(env_ids, force_reset)
        # vis_obs = np.array(torch.squeeze(obs["vis_obs"]).detach().cpu(),dtype="uint8")
        if self.eval==True:
            vis_obs = np.squeeze((obs["vis_obs"]).detach().clone().cpu().numpy()).astype("uint8")
            return [dm_env.TimeStep(step_type=dm_env.StepType.FIRST, 
                                reward=None,
                                discount=1.0,
                                observation=vis_obs)]
        else:
            vis_obs_batch = (obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")
            return [dm_env.TimeStep(step_type=dm_env.StepType.FIRST, 
                               reward=None,
                               discount=1.0,
                               observation=vis_obs) for vis_obs in vis_obs_batch]
         
    
    def step(self, action):
        obs, rew_batch, done_batch, extra_info = self.env.step(torch.tanh(torch.tensor(action, dtype = torch.float32, device = self.device)))
        del extra_info
        self.raw_rew[:] = rew_batch.detach().clone().cpu().numpy()
        if self.eval==True:
            vis_obs = np.squeeze((obs["vis_obs"]).detach().clone().cpu().numpy()).astype("uint8")
            return [dm_env.TimeStep(step_type=dm_env.StepType.MID if not done_batch else dm_env.StepType.LAST,
                               reward=rew_batch.detach().clone().cpu().item(),
                               discount=1.0,
                               observation=vis_obs)]
        else:
            vis_obs_batch = (obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")
            return [dm_env.TimeStep(step_type=dm_env.StepType.MID if not done else dm_env.StepType.LAST,
                                reward=rew.detach().clone().cpu().item(),
                                discount=1.0,
                                observation=vis_obs) for (vis_obs, rew, done) in zip(vis_obs_batch, rew_batch, done_batch)]
    
    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def action_spec(self):
        return self._action_spec
    
    def discount_spec(self):
        return self._discount_spec
    
    def render(self):
        mujoco_joint_q = self.env.get_mujoco_joint_q(self.env.state.joint_q.view(self.env.num_envs, -1)[0]).detach().cpu().numpy()
        # self.dmc_render.render(mujoco_joint_q, self.render_kwargs)
        # self.env.render(mujoco_joint_q)
        return self.dmc_render.render(mujoco_joint_q, self.render_kwargs) # since we only have one env, so the envid is 0

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

    def setup(self):
        # create logger
        # self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        # env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        train_env = MakeDMfromShac(self.cfg, False)
        eval_env = MakeDMfromShac(self.cfg, True)
        self.env = train_env
        self.train_env = dmc.make_from_shac(train_env, self.cfg, False)
        self.eval_env = dmc.make_from_shac(eval_env, self.cfg, True)
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


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset(env_ids = None, force_reset = True)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step[0].last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step[0].observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step[0].reward
                step += 1
            episode += 1
            self.video_recorder.save(f'{self.step_count}.mp4')

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
                # import pdb; pdb.set_trace()
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
        self._global_episode += len(done_ids)

    def train(self):
        # predicates
        self.start_time = time.time()
        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        
        self.time_report.start_timer("algorithm")

        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames * self.num_envs,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_steps = self.train_env.reset(env_ids = None, force_reset = True)
        self.process_time_steps(time_steps)

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.step_count):
                self.eval()

            # sample action
            # output action with shape (num_envs, action_size)
            with torch.no_grad(), utils.eval_mode(self.agent):
                self.vis_obs_buffer[:] = torch.tensor([time_step.observation for time_step in time_steps])
                action = self.agent.act(self.vis_obs_buffer,
                                        self.global_step,
                                        eval_mode=False)
            
            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.save_snapshot()

            # take env step       
            time_steps = self.train_env.step(action)
            episode_reward += np.sum([time_step.reward for time_step in time_steps])
            self.process_time_steps(time_steps)
            self.step_count += self.num_envs * self.cfg.action_repeat
            episode_step += 1

            self._global_step += self.num_envs
            
            # logging
            time_elapse = time.time() - self.start_time
            if (len(self.episode_loss_his) > 0):
                mean_policy_loss = self.episode_loss_meter.get_mean()
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, episode_step)

            self.writer.flush()
        
        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.close()

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