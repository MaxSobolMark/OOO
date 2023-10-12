import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.exploration_agent_config = ml_collections.ConfigDict()

    config.exploration_agent_config.actor_lr = 3e-4
    config.exploration_agent_config.value_lr = 3e-4
    config.exploration_agent_config.critic_lr = 3e-4

    config.exploration_agent_config.hidden_dims = (256, 256)

    config.exploration_agent_config.discount = 0.99

    config.exploration_agent_config.expectile = 0.8  # The actual tau for expectiles.
    config.exploration_agent_config.temperature = 3.0
    config.exploration_agent_config.dropout_rate = 0.1

    config.exploration_agent_config.tau = 0.005  # For soft target updates.

    config.exploration_agent_config.opt_decay_schedule = (
        None  # Don't decay optimizer lr
    )
    config.exploration_agent_config.bound_q_functions = True
    config.exploration_agent_config.min_reward = -1.0
    config.exploration_agent_config.max_reward = (
        5.0  # Very loose bound, only to stabilize training.
    )

    return config
