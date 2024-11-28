import numpy as np
from scipy import sparse
import numba


@numba.jit(nopython=True)
def _value_iteration_step(P_data, P_indices, P_indptr, R_data, R_indices, R_indptr,
                          V, discount, n_states, n_actions):
    """
    Optimized single step of value iteration using Numba
    """
    V_new = np.zeros_like(V)

    for s in range(n_states):
        max_value = float('-inf')

        for a in range(n_actions):
            value = 0

            # Get the slice for current state-action pair
            start = P_indptr[s + a * n_states]
            end = P_indptr[s + a * n_states + 1]

            # Calculate value for this action
            for idx in range(start, end):
                next_state = P_indices[idx]
                prob = P_data[idx]
                reward = R_data[idx]
                value += prob * (reward + discount * V[next_state])

            max_value = max(max_value, value)

        V_new[s] = max_value

    return V_new


def optimized_value_iteration(P, R, discount=0.95, epsilon=0.01, max_iter=1000):
    """
    Optimized value iteration for MDPs with sparse transition matrices

    Parameters:
    -----------
    P : numpy.ndarray of scipy.sparse.csr_matrix
        Array of sparse transition matrices, one for each action
    R : numpy.ndarray of scipy.sparse.csr_matrix
        Array of sparse reward matrices, one for each action
    discount : float
        Discount factor (default: 0.95)
    epsilon : float
        Convergence threshold (default: 0.01)
    max_iter : int
        Maximum number of iterations (default: 1000)

    Returns:
    --------
    tuple: (V, policy, n_iter)
        V : ndarray - Optimal value function
        policy : ndarray - Optimal policy
        n_iter : int - Number of iterations performed
    """
    # Convert numpy arrays of sparse matrices to lists
    P = list(P)
    R = list(R)

    n_states = P[0].shape[0]
    n_actions = len(P)

    # Convert transition matrices to single sparse matrix
    P_combined = sparse.vstack(P).tocsr()
    R_combined = sparse.vstack(R).tocsr()

    # Initialize value function
    V = np.zeros(n_states)

    # Main iteration loop with early stopping
    for n in range(max_iter):
        V_new = _value_iteration_step(
            P_combined.data, P_combined.indices, P_combined.indptr,
            R_combined.data, R_combined.indices, R_combined.indptr,
            V, discount, n_states, n_actions
        )

        # Check convergence
        delta = np.abs(V - V_new).max()
        V = V_new

        if delta < epsilon:
            break

    # Compute optimal policy
    policy = np.zeros(n_states, dtype=int)
    Q = np.zeros((n_actions, n_states))

    for a in range(n_actions):
        # Get immediate rewards
        immediate_rewards = np.array(R[a].sum(axis=1)).flatten()
        # Get expected future values
        expected_future = P[a].dot(V)
        # Combine immediate rewards and discounted future values
        Q[a] = immediate_rewards + discount * expected_future

    policy = np.argmax(Q, axis=0)

    return V, policy, n + 1