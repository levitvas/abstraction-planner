import time
import numpy as np
import torch


class FastValueIteration:
    def __init__(self, transition_matrix, reward_matrix, gamma=0.99, epsilon=0.01, max_iter=1000, device='cuda', prev_V=None):
        self.policy = None
        self.device = device if torch.cuda.is_available() else 'cpu'
        # print(self.device)

        # Convert scipy CSR to torch CSR
        self.P = [torch.sparse_csr_tensor(
            t.indptr, t.indices, t.data, t.shape, device=self.device, dtype=torch.float32
        ) for t in transition_matrix]

        self.R = [torch.sparse_csr_tensor(
            r.indptr, r.indices, r.data, r.shape, device=self.device, dtype=torch.float32
        ) for r in reward_matrix]

        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.S = transition_matrix[0].shape[0]
        self.A = len(transition_matrix)
        if prev_V is not None:
            self.V = torch.tensor(prev_V, dtype=torch.float32, device=self.device)
        else:
            self.V = torch.zeros(self.S, dtype=torch.float32, device=self.device)
        # self.R_dense = torch.stack([r.to_dense() for r in self.R])
        # self.P_batch = torch.stack([p.to_dense() for p in self.P])
        # Precompute expected rewards
        self.expected_rewards = []
        for a in range(self.A):
            P_a_coo = self.P[a].to_sparse_coo()
            R_a_coo = self.R[a].to_sparse_coo()

            P_indices = P_a_coo.indices()
            P_values = P_a_coo.values()
            R_values = R_a_coo.values()

            PR_values = P_values * R_values

            expected_reward = torch.zeros(self.S, device=self.device)
            rows = P_indices[0]
            expected_reward.scatter_add_(0, rows, PR_values)

            self.expected_rewards.append(expected_reward)


    def bellman_operator(self):
        Q = torch.zeros((self.A, self.S), device=self.device)

        for a in range(self.A):
            # Use precomputed expected reward
            expected_reward = self.expected_rewards[a]

            # Compute expected future value using sparse matrix-vector product
            expected_future_value = self.gamma * (self.P[a] @ self.V)

            # Combine to get Q-values
            Q[a] = expected_reward + expected_future_value

        # Find the best action for each state
        self.V, self.policy = torch.max(Q, dim=0)

        return self.V, self.policy

    def run(self):
        start_time = time.time()
        stats = []

        for i in range(self.max_iter):
            V_prev = self.V.clone()
            self.V, self.policy = self.bellman_operator()

            diff = torch.max(torch.abs(self.V - V_prev)).item()
            stats.append({
                'iteration': i + 1, 'diff': diff,
                'value_mean': self.V.mean().item(),
                'time': time.time() - start_time
            })

            if diff < self.epsilon:
                break

        return self.V.cpu().numpy(), self.policy.cpu().numpy()