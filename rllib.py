import ray
from ray import tune

ray.init(address="ray://example-ray-head:10001")
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "num_workers": 10,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
)

