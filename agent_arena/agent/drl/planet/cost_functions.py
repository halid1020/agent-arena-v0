import torch.distributions as td
from agent_arena.agent.utilities.torch_utils import *


def trajectory_return(trajectory, agent):
    planning_horizon = trajectory['deter'].shape[0]
    #print('trajectory', trajectory['deter'].shape)
    returns = agent.model['reward_model'](
        trajectory['deter']\
            .view(-1, agent.config.deterministic_latent_dim), 
        trajectory['stoch']['sample']\
            .view(-1, agent.config.stochastic_latent_dim))\
            .view(planning_horizon, -1).sum(dim=0)
    #print('returns shape', returns.shape)
    return -returns.detach().cpu().numpy()

def last_step_z_divergence_goal(trajectory, goal, agent, revserse=False):


    planning_horizon = trajectory['deter'].shape[0]
    last_step_mean = trajectory['stoch']['mean']\
            .view(planning_horizon, -1, agent.config.stochastic_latent_dim)[-1:]
    
    last_step_std = trajectory['stoch']['std']\
            .view(planning_horizon, -1, agent.config.stochastic_latent_dim)[-1:]
    
    
    ## Get Goal Distribution
    goal_image = agent.transform(
            {agent.config.input_obs: np.expand_dims(goal[agent.config.input_obs], axis=(0, 1)).transpose(0, 1, 4, 2, 3)}, 
            train=False)[agent.config.input_obs]
    
    goal_state = {
        'deter': torch.zeros(
            1, agent.config.deterministic_latent_dim, 
            device=agent.config.device),
        'stoch': {
            'sample': torch.zeros(
                1, agent.config.stochastic_latent_dim, 
                device=agent.config.device)
        },
        'input_obs': goal_image # batch*horizon*C*H*W
    }
    
    
    action = np_to_ts(np.asarray(agent.config.no_op), agent.config.device).unsqueeze(0).unsqueeze(0) # B*H*action_dim
    goal_state = agent.unroll_state_action_(goal_state, action)

    # extrat goal mean and std to match last step dim
    goal_mean = goal_state['stoch']['mean'].repeat(1, last_step_mean.shape[1], 1)
    goal_std = goal_state['stoch']['std'].repeat(1, last_step_mean.shape[1], 1)
    
    # print('goal_state', goal_mean.shape)
    # print('last_step_mean', last_step_mean.shape)

    
    goal_distribution = ContDist(td.independent.Independent(
                td.normal.Normal(goal_mean, goal_std), 1))._dist
    
    last_step_distribution = ContDist(td.independent.Independent(
                td.normal.Normal(last_step_mean, last_step_std), 1))._dist
    
    if revserse:
        divergence = td.kl.kl_divergence(goal_distribution, last_step_distribution)\
                        .detach().cpu().squeeze(0).numpy()
    else:
        divergence = td.kl.kl_divergence(last_step_distribution, goal_distribution)\
                            .detach().cpu().squeeze(0).numpy()
    
    #print('divergence', divergence.shape)

    return divergence


def last_step_z_distance_goal_stoch(trajectory, goal, agent):


    planning_horizon = trajectory['deter'].shape[0]
    last_step_mean = trajectory['stoch']['mean']\
            .view(planning_horizon, -1, agent.config.stochastic_latent_dim)[-1]
    
    
    
    ## Get Goal Distribution
    goal_image = agent.transform(
            {agent.config.input_obs: np.expand_dims(goal[agent.config.input_obs], axis=(0, 1)).transpose(0, 1, 4, 2, 3)}, 
            train=False)[agent.config.input_obs]
    
    goal_state = {
        'deter': torch.zeros(
            1, agent.config.deterministic_latent_dim, 
            device=agent.config.device),
        'stoch': {
            'sample': torch.zeros(
                1, agent.config.stochastic_latent_dim, 
                device=agent.config.device)
        },
        'input_obs': goal_image # batch*horizon*C*H*W
    }
    
    
    action = np_to_ts(np.asarray(agent.config.no_op), agent.config.device).unsqueeze(0).unsqueeze(0) # B*H*action_dim
    goal_state = agent.unroll_state_action_(goal_state, action)

    # extrat goal mean and std to match last step dim
    goal_mean = goal_state['stoch']['mean'].repeat(1, last_step_mean.shape[0], 1)[0]

    # print('goal_state', goal_mean.shape)
    # print('last_step_mean', last_step_mean.shape)


    return torch.norm(goal_mean - last_step_mean, dim=-1).detach().cpu().numpy()
    
def last_step_z_distance_goal_deter(trajectory, goal, agent):


    planning_horizon = trajectory['deter'].shape[0]
    last_step = trajectory['deter']\
            .view(planning_horizon, -1, agent.config.deterministic_latent_dim)[-1]
    
    
    
    ## Get Goal Distribution
    goal_image = agent.transform(
            {agent.config.input_obs: np.expand_dims(goal[agent.config.input_obs], axis=(0, 1)).transpose(0, 1, 4, 2, 3)}, 
            train=False)[agent.config.input_obs]
    
    goal_state = {
        'deter': torch.zeros(
            1, agent.config.deterministic_latent_dim, 
            device=agent.config.device),
        'stoch': {
            'sample': torch.zeros(
                1, agent.config.stochastic_latent_dim, 
                device=agent.config.device)
        },
        'input_obs': goal_image # batch*horizon*C*H*W
    }
    
    
    action = np_to_ts(np.asarray(agent.config.no_op), agent.config.device).unsqueeze(0).unsqueeze(0) # B*H*action_dim
    goal_state = agent.unroll_state_action_(goal_state, action)

    # extrat goal mean and std to match last step dim
    goal = goal_state['deter'].repeat(1, last_step.shape[0], 1)[0]

    # print('goal_state', goal_mean.shape)
    # print('last_step_mean', last_step_mean.shape)


    return torch.norm(goal - last_step, dim=-1).detach().cpu().numpy()


def last_step_z_distance_goal_both(trajectory, goal, agent):


    planning_horizon = trajectory['deter'].shape[0]
    last_step_deter = trajectory['deter']\
            .view(planning_horizon, -1, agent.config.deterministic_latent_dim)[-1]

    last_step_stoch = trajectory['stoch']['mean']\
            .view(planning_horizon, -1, agent.config.stochastic_latent_dim)[-1]
    
    
    
    ## Get Goal Distribution
    goal_image = agent.transform(
            {agent.config.input_obs: np.expand_dims(goal[agent.config.input_obs], axis=(0, 1)).transpose(0, 1, 4, 2, 3)}, 
            train=False)[agent.config.input_obs]
    
    goal_state = {
        'deter': torch.zeros(
            1, agent.config.deterministic_latent_dim, 
            device=agent.config.device),
        'stoch': {
            'sample': torch.zeros(
                1, agent.config.stochastic_latent_dim, 
                device=agent.config.device)
        },
        'input_obs': goal_image # batch*horizon*C*H*W
    }
    
    
    action = np_to_ts(np.asarray(agent.config.no_op), agent.config.device).unsqueeze(0).unsqueeze(0) # B*H*action_dim
    goal_state = agent.unroll_state_action_(goal_state, action)

    # extrat goal mean and std to match last step dim
    goal_deter = goal_state['deter'].repeat(1, last_step_deter.shape[0], 1)[0]
    goal_stoch = goal_state['stoch']['mean'].repeat(1, last_step_stoch.shape[0], 1)[0]

    goal = torch.cat([goal_deter, goal_stoch], dim=-1)
    last_step = torch.cat([last_step_deter, last_step_stoch], dim=-1)

    # print('goal_state', goal_mean.shape)
    # print('last_step_mean', last_step_mean.shape)


    return torch.norm(goal - last_step, dim=-1).detach().cpu().numpy()