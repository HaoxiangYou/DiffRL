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

import dm_env
from dm_env import specs
import envs
from externals.drqv2 import utils
from externals.drqv2 import dmc
from externals.drqv2.logger import Logger
from externals.drqv2.replay_buffer import ReplayBufferStorage, make_replay_loader
from externals.drqv2.video import TrainVideoRecorder, VideoRecorder
from utils.common import *
from viewer.dmc_viewer import DMCViewer

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class MakeDMfromShac(dm_env.Environment):
    def __init__(self, cfg):
        self.cfg = cfg
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        seeding(cfg["params"]["general"]["seed"])
        self.env =  env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
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
        self.render_size = 256 # fixed due to the data mismatch with TrainVideoRecorder
        self.camera_id = 0 # render camera id. 
        self.render_kwargs = dict(height=self.render_size, width=self.render_size, camera_id=self.camera_id)
        self.dmc_render = DMCViewer(file_path=os.path.join(project_dir, "envs/assets/half_cheetah.xml"), 
                                            camera_id=0, height=self.render_size, width=self.render_size)
        
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
            self._action_spec = specs.BoundedArray((self.env.num_actions,),
                                                   minimum=-1,
                                                   maximum=1,
                                                   dtype='float32',
                                                   name='action')
        self._reward_spec = specs.Array(shape=(), dtype=np.dtype('float32'), name='reward')
        self._discount_spec = specs.BoundedArray(
        shape=(), dtype='float32', minimum=0.0, maximum=1.0, name='discount')
        if hasattr(self.env, 'discount_spec'):
            self._discount_spec = self.env.discount_spec()

    def reset(self):
        # return stacked observation (9 * width * height)
        self.env.clear_grad()
        obs = self.env.reset()
        # vis_obs = np.array(torch.squeeze(obs["vis_obs"]).detach().cpu(),dtype="uint8")
        vis_obs = np.squeeze((obs["vis_obs"]).detach().cpu().numpy()).astype("uint8")
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, 
                               reward=None,
                               discount=1.0,
                               observation=vis_obs)
    
    def step(self, action):
        obs, rew, done, extra_info = self.env.step(torch.tanh(torch.tensor(action, dtype = torch.float32, device = self.cfg.device)))
        del extra_info
        vis_obs = np.squeeze((obs["vis_obs"]).detach().cpu().numpy()).astype("uint8")
        return dm_env.TimeStep(step_type=dm_env.StepType.MID if not done else dm_env.StepType.LAST,
                               reward=rew.detach().cpu().item(),
                               discount=1.0,
                               observation=vis_obs)
    
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
        return self.dmc_render.render(mujoco_joint_q, self.render_kwargs) # since we only have one env, so the envid is 0

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        # env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        env = MakeDMfromShac(self.cfg)
        self.env = env
        self.train_env = dmc.make_from_shac(env, self.cfg)
        self.eval_env = dmc.make_from_shac(env, self.cfg)
        # print("observation_spec: ", self.train_env.observation_spec())
        # print("action_spec: ", self.train_env.action_spec())
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
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
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)
            
            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

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