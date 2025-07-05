import argparse
import os
import ray
import agent_arena as ag_ar
from ruamel.yaml import YAML
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel_training_config', default='example.yaml')
    parser.add_argument('--log_dir', default='ray_results')
    parser.add_argument('--verbose', default='info')
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    parser.add_argument('--arenas_per_agent', default=8, type=int)
    parser.add_argument('--local', action="store_true")
    return parser.parse_args()

@ray.remote(num_gpus=0.5)
class AgentTrainer:
    def __init__(self, agent_id, training_config, log_dir, disp, arenas_per_agent):
        self.agent_id = agent_id
        self.config = ag_ar.retrieve_config(training_config['agent'], training_config['arena'], training_config['config'])
        self.agent = ag_ar.build_agent(training_config['agent'], config=self.config)
        save_dir = os.path.join(log_dir, training_config['arena'], training_config['agent'], training_config['config'], f"agent_{agent_id}")
        
        self.arena = ag_ar.build_arena(f"{training_config['arena']},disp:0")
        
        self.agent.set_log_dir(save_dir)
        self.arena.set_log_dir(save_dir)
        self.validation_interval = self.config.validation_interval
        self.total_update_steps = self.config.total_update_steps
        

    def train(self):
        ag_ar.train_and_evaluate(self.agent, self.arena,
            self.validation_interval, self.total_update_steps, -1)
        return f"Agent {self.agent_id} finished training."

def main():
    args = parse_arguments()

    if args.local:
        ray.init(local_mode=True)
    else:
        ray.init(address='auto', namespace='multi-agent-train')  # Assumes Ray cluster is up
    

    # Load YAML config file using ruamel.yaml
    yaml = YAML(typ='safe', pure=True)
    config_path = Path(args.parallel_training_config)
    config_data = yaml.load(config_path.read_text())
    training_configs = config_data['training_configs']

    # Check that the number of agents matches the number of configs
    num_train = len(training_configs)
    # Create Ray actors, each getting its own config
    trainers = [
        AgentTrainer.remote(
            agent_id=i,
            training_config=training_configs[i],
            log_dir=args.log_dir,
            disp=args.disp,
            arenas_per_agent=args.arenas_per_agent
        )
        for i in range(num_train)
    ]

    # Kick off training in parallel
    futures = [trainer.train.remote() for trainer in trainers]
    results = ray.get(futures)
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
