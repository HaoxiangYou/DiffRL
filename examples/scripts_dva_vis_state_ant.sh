seeds='100 200 300 400 500'
algorithms='dva'
envs='ant'   
for env in $envs
do
    for algorithm in $algorithms
    do 
        for seed in $seeds
        do
            echo "Now Running Experiment: $algorithm, Robot: $env Seed: $seed"
            python train_dva.py --cfg ./cfg/$algorithm/$env.yaml --logdir ./logs/$env/$algorithm --seed $seed
        done   
    done
done 


