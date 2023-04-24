import ray
from ray import tune

#ray.init(address="ray://ibmcloud-ray-head:10001")
ray.init(address="ray://example-ray-head:10001")
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_envs_per_worker": 10,
        "num_gpus_per_worker": 0.2,
        "framework": "torch",
        "num_workers": 2,
        "lr": tune.grid_search([0.001, 0.0001,0.005])
    },
)

