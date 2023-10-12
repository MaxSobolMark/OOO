import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.exploration_agent_config = ml_collections.ConfigDict()
    config.exploration_agent_config.td3_update = True
    config.exploration_agent_config.actor_lr = 3e-4
    config.exploration_agent_config.value_lr = 3e-4
    config.exploration_agent_config.critic_lr = 3e-4

    config.exploration_agent_config.hidden_dims = (256, 256)

    config.exploration_agent_config.discount = 0.99

    config.exploration_agent_config.expectile = 0.7  # The actual tau for expectiles.
    config.exploration_agent_config.temperature = 0.5
    config.exploration_agent_config.dropout_rate = 0.1

    config.exploration_agent_config.tau = 0.005  # For soft target updates.

    config.exploration_agent_config.opt_decay_schedule = (
        None  # Don't decay optimizer lr
    )

    return config
