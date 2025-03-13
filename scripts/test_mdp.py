import numpy as np
from scipy.sparse import csr_matrix

from utils.gpu_value import FastValueIteration


def create_test_grid_world(size=4):
    """Create a simple grid world MDP with deterministic transitions"""
    S = size * size
    A = 4  # up, down, left, right

    # Initialize sparse matrices for transitions and rewards
    transition_matrices = []
    reward_matrices = []

    # Define actions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for a in range(A):
        # Initialize data for sparse matrices
        data, row_indices, col_indices = [], [], []
        reward_data, reward_row, reward_col = [], [], []

        for s in range(S):
            row, col = s // size, s % size

            # Check if goal state (top-right corner)
            if row == 0 and col == size - 1:
                # Self-loop at goal with zero reward
                data.append(1.0)
                row_indices.append(s)
                col_indices.append(s)
                reward_data.append(0.0)
                reward_row.append(s)
                reward_col.append(s)
                continue

            # Compute next state based on action
            new_row = max(0, min(size - 1, row + directions[a][0]))
            new_col = max(0, min(size - 1, col + directions[a][1]))
            next_s = new_row * size + new_col

            # Add transition
            data.append(1.0)
            row_indices.append(s)
            col_indices.append(next_s)

            # Add reward (-1 for each step, 10 for reaching goal)
            reward = 10 if next_s == (size - 1) else -1
            reward_data.append(float(reward))
            reward_row.append(s)
            reward_col.append(next_s)

        # Create sparse matrices
        transition = csr_matrix((data, (row_indices, col_indices)), shape=(S, S))
        reward = csr_matrix((reward_data, (reward_row, reward_col)), shape=(S, S))

        transition_matrices.append(transition)
        reward_matrices.append(reward)

    return transition_matrices, reward_matrices


def reference_value_iteration(P, R, gamma=0.99, epsilon=0.01, max_iter=1000):
    """Standard value iteration implementation for verification"""
    S = P[0].shape[0]
    A = len(P)

    V = np.zeros(S)
    policy = np.zeros(S, dtype=int)

    for i in range(max_iter):
        V_prev = V.copy()

        for s in range(S):
            Q_values = np.zeros(A)

            for a in range(A):
                # Get transition probabilities and rewards for state s and action a
                p_row = P[a][s].toarray().flatten()
                r_row = R[a][s].toarray().flatten()

                # Calculate Q(s,a)
                Q_values[a] = np.sum(p_row * (r_row + gamma * V_prev))

            # Update value function and policy
            V[s] = np.max(Q_values)
            policy[s] = np.argmax(Q_values)

        # Check convergence
        if np.max(np.abs(V - V_prev)) < epsilon:
            break

    return V, policy


def test_value_iteration():
    # Create test MDP
    P, R = create_test_grid_world(size=4)

    # Run your fast implementation
    vi_fast = FastValueIteration(P, R, gamma=0.99, epsilon=0.0001)
    V_fast, policy_fast = vi_fast.run()

    # Run reference implementation
    V_ref, policy_ref = reference_value_iteration(P, R, gamma=0.99, epsilon=0.01)

    # Compare results
    value_diff = np.max(np.abs(V_fast - V_ref))
    policy_match = np.mean(policy_fast == policy_ref) * 100

    print(f"Maximum value difference: {value_diff:.6f}")
    print(f"Policy match percentage: {policy_match:.2f}%")

    # Visualize results (for a grid world)
    print("\nOptimal Values:")
    visualize_grid_values(V_fast, size=4)

    print("\nOptimal Values 2:")
    visualize_grid_values(V_ref, size=4)

    print("\nOptimal Policy:")
    visualize_grid_policy(policy_fast, size=4)

    # Return True if implementations match
    return value_diff < 1e-4 and policy_match > 99.0


def verify_bellman_equation(P, R, V, policy, gamma=0.99):
    """Verify that the Bellman optimality equation holds"""
    S = len(V)
    max_violation = 0.0

    for s in range(S):
        # Get optimal action
        a = policy[s]

        # Calculate expected value using the Bellman equation
        p_row = P[a][s].toarray().flatten()
        r_row = R[a][s].toarray().flatten()
        expected_v = np.sum(p_row * (r_row + gamma * V))

        # Check difference
        violation = abs(V[s] - expected_v)
        max_violation = max(max_violation, violation)

    print(f"Maximum Bellman equation violation: {max_violation:.6f}")
    return max_violation < 1e-4


def visualize_grid_values(V, size):
    """Visualize value function as a grid"""
    for i in range(size):
        for j in range(size):
            s = i * size + j
            print(f"{V[s]:7.2f}", end=" ")
        print()


def visualize_grid_policy(policy, size):
    """Visualize policy as arrows"""
    arrows = ["↑", "↓", "←", "→"]
    for i in range(size):
        for j in range(size):
            s = i * size + j
            print(f" {arrows[policy[s]]} ", end="")
        print()


if __name__ == "__main__":
    success = test_value_iteration()

    if success:
        print("\nTests PASSED: Your value iteration implementation is correct!")
    else:
        print("\nTests FAILED: There may be issues in your implementation.")
