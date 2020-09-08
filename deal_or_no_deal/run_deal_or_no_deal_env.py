import gym
import torch

from deal_or_no_deal.external.deep_rl.agents.DQN_agents.DQN import DQN
from deal_or_no_deal.external.deep_rl.agents.Trainer import Trainer
from deal_or_no_deal.external.deep_rl.utilities.data_structures.Config import Config


config = Config()
config.seed = 42

config.environment = gym.make('deal-or-no-deal-v0')

config.num_episodes_to_run = 1000
config.file_to_save_data_results = '../data/deal_or_no_deal.pkl'
config.file_to_save_results_graph = '../data/deal_or_no_deal.png'
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = torch.cuda.is_available()
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    'DQN_Agents': {
        'linear_hidden_units': [16, 8],
        'learning_rate': 0.01,
        'buffer_size': 40000,
        'batch_size': 256,
        'final_layer_activation': 'None',
        'columns_of_data_to_be_embedded': [0],
        'embedding_dimensions': [[29, 10]],
        'batch_norm': False,
        'gradient_clipping_norm': 5,
        'update_every_n_steps': 1,
        'epsilon_decay_rate_denominator': 10,
        'discount_rate': 0.99,
        'learning_iterations': 1,
        'tau': 0.01,
        'exploration_cycle_episodes_length': None,
        'learning_iterations': 1,
        'clip_rewards': False,
    },
}

if __name__ == '__main__':
    AGENTS = [DQN]
    trainer = Trainer(config, AGENTS)

    trainer.run_games_for_agents()
