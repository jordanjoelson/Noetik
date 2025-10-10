import torch
from iql import IQLConfig, IQLAgent, Batch
from iql.utils import preprocess_kuairand


def main():
    # Load and preprocess the dataset
    dataset = preprocess_kuairand("data/kuairand_1k.csv")

    print("Loaded dataset:")
    print(f"obs shape: {dataset.obs.shape}")
    print(f"act shape: {dataset.act.shape}")
    print(f"rew shape: {dataset.rew.shape}")
    print(f"next_obs shape: {dataset.next_obs.shape}")
<<<<<<< HEAD
    print(f"done shape: {dataset.done.shape}") 
=======
    print(f"done shape: {dataset.done.shape}")
>>>>>>> 15a45c0 (starter model files)

    # Configure and build the agent
    obs_dim = dataset.obs.shape[1]
    act_dim = dataset.act.shape[1]
    cfg = IQLConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(256, 256),
        batch_size=256,
    )
    agent = IQLAgent(cfg)

    # Training loop
    num_steps = 1000
    for step in range(num_steps):
        idx = torch.randint(0, len(dataset.obs), (cfg.batch_size,))
        batch = Batch(
            dataset.obs[idx],
            dataset.act[idx],
            dataset.rew[idx],
            dataset.next_obs[idx],
            dataset.done[idx],
        )

        logs = agent.update(batch)

        if step % 50 == 0:
            print(f"Step {step}: ", {k: round(v, 4) for k, v in logs.items()})

    # Test deterministic policy actions
    with torch.no_grad():
        test_obs = dataset.obs[:5]
        actions = agent.act(test_obs, deterministic=True)
        print("Sample deterministic actions:\n", actions)


if __name__ == "__main__":
    main()
