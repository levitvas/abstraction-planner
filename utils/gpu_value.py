import time
import numpy as np
import torch


class FastValueIteration:
    def __init__(self, transition_matrix, reward_matrix, gamma=0.99, epsilon=0.01, max_iter=1000, device='cuda'):
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
        self.V = torch.zeros(self.S, dtype=torch.float32, device=self.device)
        self.R_dense = torch.stack([r.to_dense() for r in self.R])


    def bellman_operator(self):
        # Stack sparse matrices for batch operation
        P_batch = torch.stack([p.to_dense() for p in self.P])
        future_values = torch.matmul(P_batch, self.V)
        Q = torch.sum(P_batch * (self.R_dense + self.gamma * self.V), dim=2)
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