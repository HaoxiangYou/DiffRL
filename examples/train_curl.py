import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import numpy as np
import torch
import argparse
import os
import math
import sys
import random
import time
import json
import dmc2gym
import copy
import yaml
import envs
from viewer.dmc_viewer import DMCViewer
import externals.curl.utils as utils
from externals.curl.logger import Logger
from externals.curl.video import VideoRecorder

from externals.curl.curl_sac import CurlSacAgent
from torchvision import transforms
import dm_env
from gymnasium import core, spaces


class GymEnvWrapperfromdFlex(core.Env):
    def __init__(self, env, cfg):
        self.env = env
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions
        self.render_size = 256 # fixed due to the data mismatch with TrainVideoRecorder
        self.camera_id = 0 # render camera id. 
        self.render_kwargs = dict(height=self.render_size, width=self.render_size, camera_id=self.camera_id)
        self.dmc_render_model = cfg["params"]["general"]["dmc_render_model"]
        self.dmc_render = DMCViewer(file_path=os.path.join(project_dir, f"envs/assets/{self.dmc_render_model}.xml"), 
                                            camera_id=0, height=self.render_size, width=self.render_size)
        self.raw_rew = np.zeros((self.env.num_envs)) 
        self.device = cfg["params"]["general"]["device"]

        self._observation_space = spaces.Box(shape=self.env.num_vis_obs,
                                                low= 0,
                                                high= 255,
                                                dtype='uint8')
        # overwrite action space for parallel computation
        self._action_space = spaces.Box(shape=(self.num_envs, self.env.num_actions),
                                                low=-1,
                                                high=1,
                                                dtype='float32')
    
    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        # return stacked observation (9 * width * height)
        self.env.clear_grad()
        obs = self.env.reset(env_ids=env_ids, 
                             force_reset=force_reset,
                             enable_vis_obs=enable_vis_obs)
        reset_obs = obs["vis_obs"].detach().clone().cpu().numpy().astype("uint8")
        return reset_obs

    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        obs, rew_batch, done_batch, extra_info = self.env.step(actions = torch.tanh(torch.tensor(actions, dtype = torch.float32, device = self.device)), 
                                                               enable_reset = enable_reset, 
                                                               enable_vis_obs = enable_vis_obs)
        del extra_info
        self.raw_rew[:] = rew_batch.detach().clone().cpu().numpy()
        next_vis_obs = (obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")
        return next_vis_obs, rew_batch.detach().clone().cpu().numpy(), done_batch.detach().clone().cpu().numpy(), {}
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def render(self):
        mujoco_joint_q = self.env.get_mujoco_joint_q(self.env.state.joint_q.view(self.env.num_envs, -1)[0]).detach().cpu().numpy()
        return self.dmc_render.render(mujoco_joint_q, self.render_kwargs) # since we only have one env, so the envid is 0
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='.', type=str)
    parser.add_argument('--cfg', default='./cfg/curl/hopper.yaml', type=str)
    args = parser.parse_args()
    return args

def load_env(cfg, eval=False):
    env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
    env =  env_fn(num_envs = 1 if eval else cfg["params"]["general"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            img_height = cfg["params"]["general"].get("pre_transform_image_size", 100),\
                            img_width = cfg["params"]["general"].get("pre_transform_image_size", 100),\
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            no_grad = True)
    env = GymEnvWrapperfromdFlex(env, cfg)
    return env

def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
            video.save('%d.mp4' % step)
            
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    utils.set_seed_everywhere(cfg["params"]["general"]["seed"])

    # env = dmc2gym.make(
    #     domain_name=args.domain_name,
    #     task_name=args.task_name,
    #     seed=args.seed,
    #     visualize_reward=False,
    #     from_pixels=(args.encoder_type == 'pixel'),
    #     height=args.pre_transform_image_size,
    #     width=args.pre_transform_image_size,
    #     frame_skip=args.action_repeat
    # )
    
    train_env = load_env(cfg, eval=False)
    eval_env = load_env(cfg, eval=True)
    
    obs = train_env.reset(env_ids = None, force_reset = True, enable_vis_obs=True)
    action = train_env.action_space.sample()
    next_obs, reward, done, _ = train_env.step(action, enable_reset = False, enable_vis_obs = True)
    import pdb; pdb.set_trace()
    
    env.seed(args.seed)
    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
    env = utils.ActionDTypeWrapper(env, dtype=np.float32)

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.log_dir = args.log_dir + '/'  + exp_name

    utils.make_dir(args.log_dir)
    video_dir = utils.make_dir(os.path.join(args.log_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.log_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.log_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.log_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env.spec.max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
