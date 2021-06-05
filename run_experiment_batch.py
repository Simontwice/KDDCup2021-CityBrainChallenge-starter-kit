import subprocess
import random

for i in range(0,199,5):
    args = {"model_name":"models_for_him/round_"+str(i)+".h5","input_dir":"agent","output_dir":"out","sim_cfg":'cfg/simulator_small.cfg',
            "learning_rate":random.choice([0.001,0.005,0.0005]), "thread":4, "update_freq":random.choice([10,25,50,100]), "close_value":random.choice([30,50,70,100]), \
            "batch_size":random.choice([32,64,128,256,512]), "epsilon":random.choice([0.1,0.15,0.2]), "gamma":random.choice([0.85,0.9,0.92,0.95,0.98,0.99])}
    subprocess.run('python3 train_dqn_example.py --input_dir {input_dir} --output_dir {output_dir} --sim_cfg {sim_cfg} --model_name {model_name} \
                --batch_size {batch_size} --epsilon {epsilon} --epsilon {epsilon} --gamma {gamma} \
               --learning_rate {learning_rate} --thread {thread} --update_freq {update_freq} --close_value {close_value} '.format(**args).split())
    args = {"model_name":"models_for_him/round_"+str(i)+".h5","input_dir":"agent", "output_dir":"out","sim_cfg":'cfg/simulator_medium.cfg',
            "learning_rate":random.choice([0.001,0.005,0.0005]), "thread":4, "update_freq":random.choice([10,25,50,100]), "close_value":random.choice([30,50,70,100]), \
            "batch_size":random.choice([32,64,128,256,512]),"epsilon":random.choice([0.1,0.15,0.2]), "gamma":random.choice([0.85,0.9,0.92,0.95,0.98,0.99])}
    subprocess.run('python3 train_dqn_example.py --input_dir {input_dir} --output_dir {output_dir} --sim_cfg {sim_cfg} --model_name {model_name} \
                --batch_size {batch_size}  --epsilon {epsilon} --epsilon {epsilon} --gamma {gamma} \
               --learning_rate {learning_rate} --thread {thread} --update_freq {update_freq} --close_value {close_value} '.format(**args).split())

