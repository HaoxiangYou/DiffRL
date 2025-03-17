seeds='100 200 300 400 500'
algorithms='shac'
num_actors='16 32 64 128'
envs='humanoid' # ant cheetah hopper 
for env in $envs
do
    for algorithm in $algorithms
    do 
        for num_actor in $num_actors
        do 
            for seed in $seeds
            do
                echo "Now Running Experiment: $algorithm, Robot: $env Seed: $seed, Number of Actors: $num_actor "
                python train_shac.py --cfg ./cfg/$algorithm/$env.yaml --logdir ./logs/$env/$algorithm --seed $seed --num_actors $num_actor --no-time-stamp
            done
        done    
    done
done 


