from dataclasses import dataclass

@dataclass
class IQLConfig:
    obs_dim: int = 64        
    act_dim: int = 1
    
    hidden_sizes: tuple = (256, 256)
    policy_log_std_min: float = -5.0
    policy_log_std_max: float = 2.0
    
    discount: float = 0.99
    tau: float = 0.7
    temperature: float = 3.0
    
    lr_policy: float = 3e-4
    lr_q: float = 3e-4 
    lr_v: float = 3e-4
    weight_decay: float = 0.0
    
    batch_size: int = 256