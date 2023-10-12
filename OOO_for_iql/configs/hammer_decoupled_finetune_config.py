import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.exploration_agent_config = ml_collections.ConfigDict()

    config.exploration_agent_config.opt_decay_schedule = (
        None  # Don't decay optimizer lr
    )
    config.exploration_agent_config.policy_log_std_min = -2.0

    # update hyperparameters
    config.exploration_agent_config.expectile = 0.7
    config.exploration_agent_config.temperature = 3.0
    config.exploration_agent_config.critic_ensemble_size = 2
    config.exploration_agent_config.online_sample_temperature = 1.0
    config.exploration_agent_config.td3_update = False

    return config
