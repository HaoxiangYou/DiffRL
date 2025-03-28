# python train_ac.py --cfg ./cfg/ac/ant.yaml --logdir ./logs/test-new-loader/Ant/ac/20 --seed 20 --no-time-stamp
# python train_ac.py --cfg ./cfg/ac/ant.yaml --logdir ./logs/test-new-loader/Ant/ac/30 --seed 30 --no-time-stamp
# python train_ac.py --cfg ./cfg/ac/ant.yaml --logdir ./logs/test-new-loader/Ant/ac/40 --seed 40 --no-time-stamp
# python train_rl.py --cfg ./cfg/rl/ant.yaml --logdir ./logs/test-new-loader/Ant/rl/20 --seed 20 --no-time-stamp
# python train_rl.py --cfg ./cfg/rl/ant.yaml --logdir ./logs/test-new-loader/Ant/rl/30 --seed 30 --no-time-stamp
# python train_rl.py --cfg ./cfg/rl/ant.yaml --logdir ./logs/test-new-loader/Ant/rl/40 --seed 40 --no-time-stamp
python train_curl.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./logs/cartpole/curl \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 --save_video