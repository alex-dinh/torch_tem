'''
Connectivity masks for various attractor weight topologies.
- Hierarchical Embedding (HE)
- Scale-Free, Small-World (SFSW)
- Preferential Attachment (PA)
- Locally Connected (LC)
- Random Sparse (RS)
'''

import numpy as np
import torch
from scipy.special import comb
import matplotlib.pyplot as plt
import random

DEVICE = 'cpu'

def create_sfsw_mask(n_neurons=400, k=6, rewire_prob=0.2, hub_percent=0.05, hub_connect_percent=0.5,
                     random_hubs=True, seed=None):
    # Create a scale-free, small-world connectivity mask

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    mask = torch.zeros((n_neurons, n_neurons), dtype=torch.float)

    # Step 1: Initialize regular ring lattice
    for i in range(n_neurons):
        for offset in range(1, k // 2 + 1):
            j = (i + offset) % n_neurons
            mask[i][j] = 1
            j = (i - offset) % n_neurons
            mask[i][j] = 1
        # Allow self-connections
        mask[i][i] = 1

    # Step 2: Rewire connections with given probability
    for i in range(n_neurons):
        for j in range(n_neurons):
            if mask[i][j] == 1 and random.random() < rewire_prob:
                # # Remove original connection
                # mask[i][j] = 0
                # Choose a new target that's not i and not already connected
                possible_targets = list(set(range(n_neurons)) - {i} - set(torch.nonzero(mask[i]).flatten().tolist()))
                if possible_targets:
                    new_j = random.choice(possible_targets)
                    mask[i][new_j] = 1

    # Step 3: Add hub neurons
    n_hubs = int(n_neurons * hub_percent)
    n_hub_connections = int(n_neurons * hub_connect_percent)

    if random_hubs:
        hub_indices = random.sample(range(n_neurons), n_hubs)
        for hub in hub_indices:
            targets = random.sample([i for i in range(n_neurons) if i != hub], n_hub_connections)
            for target in targets:
                mask[hub][target] = 1
    else:
        hub_indices = np.linspace(0, n_neurons - 1, n_hubs, dtype=int)
        for hub in hub_indices:
            targets = random.sample([i for i in range(n_neurons) if i != hub], n_hub_connections)
            for target in targets:
                mask[hub][target] = 1

    # Compute sparsity of sfsw weights
    # total_elements = mask.numel()
    # num_zeros = (mask == 0).sum().item()
    # sparsity =  num_zeros / total_elements
    # print(f'Sparsity: {sparsity}')
    return mask.to(DEVICE)

def create_pa_mask(n_neurons, m_pa=None, seed=None):
    """
    Creates a connectivity mask for the TEM attractor with preferential attachment topology.

    Args:
        n_neurons (int): Number of neurons in the network - sum(params['n_p])
        topology_type (str): Type of network topology.
                             'sfsw': Scale-Free Small-World (default)
                             'pa': Preferential Attachment (Barabasi-Albert-like)
        k (int): For 'sfsw', the number of nearest neighbors in the initial ring lattice.
        rewire_prob (float): For 'sfsw', the probability of rewiring an edge.
        hub_percent (float): For 'sfsw', percentage of neurons designated as hubs.
        hub_connect_percent (float): For 'sfsw', percentage of other neurons a hub connects to.
        m_pa (int): For 'pa', the number of edges a new node forms when connecting to existing nodes.
                    If None for 'pa', it will be calculated as a fraction of n_neurons.
        random_hubs (bool): For 'sfsw', if True, hubs are chosen randomly.
        seed (int, optional): Seed for random number generators for reproducibility.

    Returns:
        torch.Tensor: A square connectivity mask (n_neurons x n_neurons).
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)  # Ensure numpy is also seeded if used by random_hubs or future additions

    mask = torch.zeros((n_neurons, n_neurons), dtype=torch.float)

    # --- Preferential Attachment (Barabasi-Albert-like) Logic ---
    # This implementation builds the network by adding nodes one by one
    # and connecting them preferentially to existing nodes with higher degrees.
    # It assumes an initially connected small graph.

    if m_pa is None:
        # Default m_pa to a reasonable value, e.g., to ensure average degree is around 4-8
        # A common choice for m_pa is small, typically 1 to 5.
        # Let's set a sensible default if not provided, e.g., 2 or 3 connections per new node.
        m_pa = max(1, int(n_neurons * 0.01))  # At least 1, or 1% of neurons as base for m_pa
        # If n_neurons is very small, ensure m_pa doesn't exceed n_neurons-1
        m_pa = min(m_pa, n_neurons - 1 if n_neurons > 1 else 1)

    if m_pa >= n_neurons:
        raise ValueError(f"m_pa ({m_pa}) must be less than n_neurons ({n_neurons}) for Preferential Attachment.")
    if m_pa <= 0:
        raise ValueError("m_pa must be a positive integer for Preferential Attachment.")

    # Step 1: Initialize with a small, fully connected core (m_pa initial nodes forming a clique)
    # A common practice is to start with m_pa fully connected nodes.
    initial_nodes_for_pa = m_pa  # Start with m_pa nodes in a clique
    if n_neurons == 1:
        mask[0][0] = 1
        return mask

    # Ensure we don't try to make a clique larger than n_neurons
    initial_nodes_for_pa = min(initial_nodes_for_pa, n_neurons)

    for i in range(initial_nodes_for_pa):
        for j in range(initial_nodes_for_pa):
            if i != j:
                mask[i][j] = 1
                mask[j][i] = 1  # Ensure symmetric connections
        mask[i][i] = 1  # Allow self-connections for initial nodes

    # Keep track of degrees for preferential attachment
    # We need degrees of existing nodes (excluding self-loops) for the PA probability calculation.
    degrees = [0] * n_neurons
    for i in range(initial_nodes_for_pa):
        degrees[i] = int(torch.sum(mask[i, :]) - mask[i, i])  # Degree is count of non-self connections

    # Step 2: Add remaining nodes with preferential attachment
    for new_node in range(initial_nodes_for_pa, n_neurons):
        # The probability of connecting to an existing node is proportional to its degree.
        # We need to select `m_pa` *distinct* existing nodes.
        existing_nodes = list(range(new_node))

        # If no existing nodes (shouldn't happen with initial_nodes_for_pa >= 1)
        if not existing_nodes:
            mask[new_node][new_node] = 1  # Only self-connection if no existing nodes
            continue

        # Calculate probabilities based on degrees (sum of degrees is 2*E, where E is num edges)
        # Add a small constant to degrees to ensure all nodes have a non-zero chance,
        # especially for early nodes with low degrees, and to avoid division by zero
        # if sum of degrees is 0 (e.g., in a theoretical sparse initial state).
        degrees_plus_epsilon = [d + 1e-6 for d in degrees[:new_node]]
        current_degrees_sum_plus_epsilon = sum(degrees_plus_epsilon)

        probabilities = [d_pe / current_degrees_sum_plus_epsilon for d_pe in degrees_plus_epsilon]

        targets = set()
        # Try to select m_pa distinct targets.
        # Ensure we don't try to select more targets than available existing nodes.
        num_targets_to_select = min(m_pa, len(existing_nodes))

        # Sample with replacement initially, then ensure uniqueness
        sampled_targets_list = random.choices(existing_nodes, weights=probabilities,
                                              k=num_targets_to_select * 2)  # Sample more to get distinct ones

        for target_node in sampled_targets_list:
            if len(targets) < num_targets_to_select and target_node != new_node:
                targets.add(target_node)
            if len(targets) == num_targets_to_select:
                break

        # Fallback if not enough distinct targets were found (e.g., very small network)
        if len(targets) < num_targets_to_select:
            remaining_needed = num_targets_to_select - len(targets)
            # Add random additional targets from available if preferential attachment failed to give enough
            additional_targets = list(set(existing_nodes) - targets - {new_node})
            if len(additional_targets) > remaining_needed:
                targets.update(random.sample(additional_targets, remaining_needed))
            else:
                targets.update(additional_targets)

        for target_node in targets:
            if new_node != target_node:  # A node cannot connect to itself via PA here (explicitly)
                if mask[new_node][
                    target_node] == 0:  # Only add if not already connected (prevents double counting for degree)
                    mask[new_node][target_node] = 1
                    mask[target_node][new_node] = 1  # Make connections symmetric
                    degrees[new_node] += 1
                    degrees[target_node] += 1
        mask[new_node][new_node] = 1  # Allow self-connections for all new neurons

    return mask

def create_random_sparse_mask(n_neurons, sparsity=0.95, seed=None):
    """
    Creates a random sparse connectivity mask where each neuron connects to a fixed
    number of other neurons, excluding self-connections, enforcing a desired sparsity.

    Args:
        n_neurons (int): The total number of neurons in the network.
        sparsity (float): The desired sparsity level (proportion of zeros).
                          E.g., 0.95 means 95% of connections will be zero.
                          Must be between 0 and 1.
        seed (int, optional): A seed for the random number generators for reproducibility.

    Returns:
        torch.Tensor: A 2D tensor (n_neurons x n_neurons) representing the connectivity mask.
                      mask[i][j] = 1 if there's a connection from i to j, 0 otherwise.
    """

    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1.")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive.")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    mask = torch.zeros((n_neurons, n_neurons), dtype=torch.int, device=DEVICE)
    for i in range(n_neurons):
        # Get possible connection targets (excluding self - don't want self-connections)
        possible_targets = list(range(n_neurons))
        possible_targets.remove(i)

        # Randomly select up to max_connections targets
        max_connections = int((1 - sparsity) * n_neurons)
        n_connections = min(max_connections, len(possible_targets))
        if n_connections > 0:
            targets = np.random.choice(possible_targets, size=n_connections, replace=False)
            mask[i, targets] = 1.0

    return mask


def _generate_mask_with_gaussian_profile(n_neurons, sigma, p_max_current, allow_self_connections, seed_val):
    """
    Helper function to generate a mask with a given p_max, maintaining the Gaussian profile.
    Internal function, not meant for direct external use.
    """
    # Ensure local seed for reproducibility within each iteration if seed_val is passed
    if seed_val is not None:
        # Vary the seed slightly for each call within the binary search to ensure different random samples
        # for a given p_max in different iterations, leading to more stable convergence.
        current_seed = seed_val + int(p_max_current * 100000)  # Multiply by larger factor for more distinct seeds
        random.seed(current_seed)
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
    else:
        # If no initial seed, ensure randomness for each call
        random.seed()
        torch.manual_seed(random.randint(0, 100000))
        np.random.seed(random.randint(0, 100000))

    mask = torch.zeros((n_neurons, n_neurons), dtype=torch.float)

    for i in range(n_neurons):
        for j in range(i, n_neurons):  # Iterate j from i to n_neurons-1 to ensure symmetric generation
            if i == j:
                if allow_self_connections:
                    # For self-connection, distance is 0, so probability is p_max_current
                    if random.random() < p_max_current:
                        mask[i][j] = 1
                # If not allowing self-connections, mask[i][i] remains 0
            else:
                # Calculate distance with periodic boundary conditions (e.g., neurons on a ring)
                # The distance between neuron i and j is the shortest path on the ring.
                distance = min(abs(i - j), n_neurons - abs(i - j))

                # Calculate connection probability using a Gaussian (exponential decay) profile:
                # P(d) = P_max * exp(-d^2 / (2 * sigma^2))
                prob_connection = p_max_current * np.exp(-(distance ** 2) / (2 * sigma ** 2))

                # Ensure probability is clamped between 0 and 1
                prob_connection = max(0.0, min(1.0, prob_connection))

                # Establish connection based on probability
                if random.random() < prob_connection:
                    mask[i][j] = 1
                    mask[j][i] = 1  # Ensure symmetric connection

    return mask


def create_locally_connected_mask(n_neurons, sigma, target_sparsity=None, p_max=None,
                                  allow_self_connections=True, seed=None,
                                  max_iter=50, sparsity_tolerance=0.001):
    """
    Creates a connectivity mask where connection probability follows a Gaussian profile
    with distance. Can enforce a target sparsity by internally adjusting p_max.

    Args:
        n_neurons (int): Number of neurons in the network.
        sigma (float): Standard deviation of the Gaussian. This controls how quickly
                       the connection probability falls off with distance. A smaller
                       'sigma' means more localized connections. Must be positive.
        target_sparsity (float, optional): Desired percentage of zeros in the mask (e.g., 0.95 for 95% sparsity).
                                         If provided, p_max will be ignored and determined internally via binary search.
                                         Must be between 0 and 1.
        p_max (float, optional): Maximum connection probability at distance 0. Used if target_sparsity is None.
                                 Must be between 0 and 1.
        allow_self_connections (bool): If True, neurons can connect to themselves.
                                       Their self-connection probability will be p_max.
        seed (int, optional): Seed for random number generators for reproducibility.
        max_iter (int): Maximum iterations for the binary search when target_sparsity is provided.
        sparsity_tolerance (float): The acceptable difference between the achieved sparsity
                                    and the target_sparsity for the binary search to converge.

    Returns:
        torch.Tensor: A square connectivity mask (n_neurons x n_neurons) where
                      mask[i][j] = 1 indicates a connection, and 0 otherwise.
                      The mask is symmetric (mask[i][j] == mask[j][i]).

    Raises:
        ValueError: If invalid arguments are provided or if neither target_sparsity nor p_max is given.
    """
    if not (sigma > 0):
        raise ValueError("sigma must be positive.")
    if not (n_neurons > 0):
        raise ValueError("n_neurons must be positive.")

    final_p_max = None

    if target_sparsity is not None:
        if not (0 <= target_sparsity <= 1):
            raise ValueError("target_sparsity must be between 0 and 1.")

        # Binary search for p_max
        low_p = 0.0
        high_p = 1.0

        # Keep track of the best p_max found so far in case convergence is not perfect
        best_p_max = (low_p + high_p) / 2.0
        min_sparsity_diff = float('inf')

        # Set the initial seed for the overall function, subsequent calls to _generate_mask_with_gaussian_profile
        # will use a modified seed to ensure new randomness within each binary search iteration.
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        for i in range(max_iter):
            current_p_max = (low_p + high_p) / 2.0

            # Generate mask and calculate actual sparsity
            # Pass the original seed (or a derived consistent one) to the helper function for reproducibility
            temp_mask = _generate_mask_with_gaussian_profile(n_neurons, sigma, current_p_max,
                                                             allow_self_connections, seed)

            num_zeros = (temp_mask == 0).sum().item()
            current_sparsity = num_zeros / (n_neurons * n_neurons)

            current_diff = abs(current_sparsity - target_sparsity)

            # Update best_p_max if current one is closer to target
            if current_diff < min_sparsity_diff:
                min_sparsity_diff = current_diff
                best_p_max = current_p_max
                final_p_max = best_p_max  # Store the best found p_max here

            if current_diff < sparsity_tolerance:
                # print(f"Converged at iteration {i+1} with p_max={current_p_max:.4f}, sparsity={current_sparsity:.4f}")
                break

            if current_sparsity < target_sparsity:  # Current mask is too dense, need higher sparsity -> decrease p_max
                high_p = current_p_max
            else:  # Current mask is too sparse, need lower sparsity -> increase p_max
                low_p = current_p_max
        else:  # Loop finished without converging within tolerance
            print(
                f"Warning: Binary search for target_sparsity={target_sparsity:.4f} did not converge within {max_iter} iterations.")
            print(
                f"Achieved sparsity: {current_sparsity:.4f} with p_max={current_p_max:.4f}. Using best p_max found: {best_p_max:.4f} with sparsity difference: {min_sparsity_diff:.4f}")
            final_p_max = best_p_max  # Use the best p_max found

    elif p_max is not None:
        if not (0 <= p_max <= 1):
            raise ValueError("p_max must be between 0 and 1.")
        final_p_max = p_max
    else:
        raise ValueError("Either target_sparsity or p_max must be provided.")

    # Generate the final mask using the determined or provided p_max
    # Use the original seed for the final generation to ensure reproducibility for the desired p_max
    return _generate_mask_with_gaussian_profile(n_neurons, sigma, final_p_max, allow_self_connections, seed)

def plot_mask(mask, plot_title):
    plt.imshow(mask, cmap='plasma', interpolation='none')
    plt.title(plot_title)
    plt.colorbar(label='Value')
    plt.grid(False)
    plt.show(block=False)

if __name__ == '__main__':
    # --- Example Usage ---
    n_neurons_val = 400

    # 1. Enforce 95% sparsity
    print("--- Locally Connected Network (Target Sparsity 95%) ---")
    lc_mask = create_locally_connected_mask(n_neurons=n_neurons_val, sigma=10,
                                            target_sparsity=0.95, allow_self_connections=True, seed=42)
    print(f"Mask shape: {lc_mask.shape}")

    total_elements = n_neurons_val * n_neurons_val
    num_actual_connections_95 = torch.sum(lc_mask)
    density_95 = num_actual_connections_95 / total_elements
    sparsity_95 = 1.0 - density_95

    print(f"Actual Density (including self-loops): {density_95:.4f}")
    print(f"Actual Sparsity (including self-loops): {sparsity_95:.4f} ({sparsity_95 * 100:.2f}%)")
    num_non_self_connections_95 = num_actual_connections_95 - torch.sum(torch.diag(lc_mask))
    average_degree_no_self_loops_95 = num_non_self_connections_95 / n_neurons_val
    print(f"Average Degree (excluding self-loops): {average_degree_no_self_loops_95:.2f}")
    plot_mask(lc_mask, 'Locally Connected (Gaussian) Mask\nsparsity=0.95')

    rs_mask = create_random_sparse_mask(n_neurons_val, sparsity=0.95, seed=42)
    plot_mask(rs_mask, 'Random Sparse Mask\nsparsity=0.95')

