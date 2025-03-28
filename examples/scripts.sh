seeds='100 200 300 400 500'
algorithms='dva'
num_actors='16 32 64 128'
envs='humanoid ant cheetah hopper' 
learning_rates='0.002'
for env in $envs
do
    for algorithm in $algorithms
    do 
        for num_actor in $num_actors
        do 
            for seed in $seeds
            do
                for learning_rate in $learning_rates
                do
                    echo "Now Running Experiment: $algorithm, Robot: $env Seed: $seed, Number of Actors: $num_actor, Actor Learning Rate: $learning_rate"
                    python train_dva.py --cfg ./cfg/$algorithm/$env.yaml --actor_learning_rate $learning_rate --logdir ./logs/$env/$algorithm --seed $seed --num_actors $num_actor --no-time-stamp
                done
            done
        done    
    done
done 


