import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class KuaiRandDataset(Dataset):
    def __init__(self, csv_path, obs_cols, act_cols, rew_col, next_obs_cols, done_col):
        self.data = pd.read_csv(csv_path)
        self.obs_cols = obs_cols
        self.act_cols = act_cols
        self.rew_col = rew_col
        self.next_obs_cols = next_obs_cols
        self.done_col = done_col

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        obs = torch.tensor(self.data.loc[idx, self.obs_cols].values, dtype=torch.float32)
        act = torch.tensor(self.data.loc[idx, self.act_cols].values, dtype=torch.float32)
        rew = torch.tensor(self.data.loc[idx, self.rew_col], dtype=torch.float32)
        next_obs = torch.tensor(self.data.loc[idx, self.next_obs_cols].values, dtype=torch.float32)
        done = torch.tensor(self.data.loc[idx, self.done_col], dtype=torch.float32)
        return obs, act, rew, next_obs, done
