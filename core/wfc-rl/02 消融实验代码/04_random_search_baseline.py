import random
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from Position_constrain import ConstraintViolation  # 确保这个 import 在文件顶部

import time
import torch.optim as optim
import WFC_library_undirected
import random
import numpy as np
import visualize_elastic_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import Homoge  # Assuming this is a custom library you have
from Position_constrain import ConstraintViolation
from WFC_library_undirected import create_prototypes
from tqdm import tqdm
from itertools import combinations
import copy

# Confirm device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_wfc_algorithm_monte_carlo(initial_grid, fiber_block_library, goal_K_tensor, num_trials):
    """
    Performs a Monte Carlo random search to find the best possible grid configuration.

    Args:
        initial_grid: The starting grid with all possibilities.
        fiber_block_library: The library of available prototypes.
        goal_K_tensor: The target stiffness tensor.
        num_trials: The total number of random grids to generate and evaluate.

    Returns:
        A tuple containing:
        - best_grid: The grid configuration with the highest reward found.
        - best_reward: The highest reward value achieved.
    """
    best_reward = float('-inf')
    best_grid = None
    device = goal_K_tensor.device
    grid_height, grid_width = len(initial_grid), len(initial_grid[0])

    print(f"Starting Monte Carlo Random Search with {num_trials} trials...")

    for _ in tqdm(range(num_trials), desc="Random Search Trials"):
        grid = copy.deepcopy(initial_grid)
        is_successful_trial = True

        # Keep collapsing until the grid is full
        while True:
            # Find all uncollapsed cells
            candidate_cells = [
                (r, c) for r in range(grid_height) for c in range(grid_width) if len(grid[r][c]) > 1
            ]

            if not candidate_cells:
                break  # Grid is fully collapsed

            # Randomly select a cell to collapse
            r, c = random.choice(candidate_cells)

            # Randomly select a state for that cell
            chosen_state = random.choice(grid[r][c])

            grid[r][c] = [chosen_state]

            try:
                # Propagate constraints, which is crucial for validity
                propagate_neighbors(grid, r, c, fiber_block_library)
            except ConstraintViolation:
                # This random choice led to a contradiction. Trial failed.
                is_successful_trial = False
                break

        # If the trial was successful, evaluate it
        if is_successful_trial:
            # Final check to ensure it's fully collapsed (might not be necessary but is safe)
            if not any(len(cell) > 1 for row in grid for cell in row):
                k_vector = visualize_elastic_module.calculate_stiffness_directions(
                    estimate_partial_stiffness(grid)
                )
                reward = calculate_potential(k_vector, goal_K_tensor.squeeze().cpu().numpy(), device)

                if reward > best_reward:
                    best_reward = reward
                    best_grid = grid

    print("Random Search finished.")
    if best_grid is None:
        print("Warning: Random search failed to find any valid solution.")

    return best_grid, best_reward

# ==================================================================
# Section: Text Visualization Tool (Replaces Matplotlib)
# ==================================================================
def print_grid_to_console(grid, title="Grid Configuration"):
    """
    Prints the final grid configuration to the console in text format.
    """
    print(f"\n--- {title} ---")
    if not grid or not grid[0]:
        print("[Empty Grid]")
        return

    # Find the longest state name for alignment
    max_len = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell_content = grid[r][c]
            name = ""
            if isinstance(cell_content, list) and len(cell_content) == 1:
                name = cell_content[0]
            elif isinstance(cell_content, str):
                name = cell_content
            max_len = max(max_len, len(name))

    if max_len == 0: max_len = 5  # Default padding

    # Print the grid
    for r in range(len(grid)):
        row_str = "|"
        for c in range(len(grid[0])):
            cell_content = grid[r][c]
            state = ""
            if isinstance(cell_content, list) and len(cell_content) == 1:
                state = cell_content[0]
            elif isinstance(cell_content, str):
                state = cell_content
            else:  # Uncollapsed or erroneous cell
                state = f"({len(cell_content)})" if isinstance(cell_content, list) else "???"
            row_str += f" {state.ljust(max_len)} |"
        print(row_str)
    print("-" * len(row_str))


# ==================================================================
# Section: Neural Network Model (Two-head: cell + state)
# ==================================================================
class WFC_RL_Net_Simplified(nn.Module):
    def __init__(self, grid_height, grid_width, num_states,
                 cnn_channels=128, fused_feature_dim=256):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_states = num_states
        self.cnn_channels = cnn_channels
        self.fused_feature_dim = fused_feature_dim

        # ------------------- 移除了 elastic_encoder 和 FiLM 相关层 -------------------

        # 简化的 CNN 编码器，可以直接增加其深度或宽度来提升模型容量
        self.cnn = nn.Sequential(
            nn.Conv2d(num_states, cnn_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels // 2), nn.GELU(),
            nn.Conv2d(cnn_channels // 2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU(),
            # 可以考虑增加一层来补偿被移除的 FiLM 层的表达能力
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU()
        )

        # 特征融合层（现在直接作用于 CNN 输出）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_channels, fused_feature_dim, kernel_size=1),
            nn.BatchNorm2d(fused_feature_dim), nn.GELU()
        )

        # 两个决策头保持不变
        self.cell_head = nn.Conv2d(fused_feature_dim, 1, kernel_size=1)
        self.state_head = nn.Conv2d(fused_feature_dim, num_states, kernel_size=1)

    # forward 函数签名也相应改变，不再接收 elastic_modulus
    def forward(self, grid_constraints):
        # grid_constraints: (B, H, W, S) -> (B, S, H, W)
        x_grid_spatial = grid_constraints.permute(0, 3, 1, 2).float()

        # ------------------- 移除了 FiLM 调制步骤 -------------------
        x_grid_features = self.cnn(x_grid_spatial)
        fused_output = self.fusion_conv(x_grid_features)

        # 决策头
        cell_logits = self.cell_head(fused_output).permute(0, 2, 3, 1).squeeze(-1)  # (B,H,W)
        state_logits = self.state_head(fused_output).permute(0, 2, 3, 1)  # (B,H,W,S)

        state_probs = F.softmax(state_logits, dim=-1)

        return cell_logits, state_probs


# ==================================================================
# Section: Core Algorithm and Helper Functions
# ==================================================================

def create_fiber_block_library():
    prototypes = create_prototypes()
    return {f"{proto.mesh[:-4]}_{proto.rotation}": proto for proto in prototypes}


def hybrid_reward(current_K, target_K, beta, scale=1e10):
    current_K, target_K = current_K.float(), target_K.float()
    cosine_sim = F.cosine_similarity(current_K.unsqueeze(0), target_K.unsqueeze(0))
    reward_linear = 2 * (cosine_sim - 0.241) / (1 - 0.241) - 1
    euclidean_dist = torch.norm(current_K - target_K, p=2)
    euclidean_reward = 1.0 - (euclidean_dist / scale)
    return (beta * reward_linear + (1 - beta) * euclidean_reward).item()


def calculate_potential(k_actual_np, k_target_np, device, beta=0.6, L2_scale=1e10):
    k_actual_tensor = torch.from_numpy(np.array(k_actual_np)).float().flatten().to(device)
    k_target_tensor = torch.from_numpy(np.array(k_target_np)).float().flatten().to(device)
    return hybrid_reward(k_actual_tensor, k_target_tensor, beta=beta, scale=L2_scale)


def compute_incremental_reward(current_K_estimated_np, target_K_np, prev_K_estimated_np, is_terminal, success):
    if is_terminal:
        if success:
            return calculate_potential(current_K_estimated_np, target_K_np, device), current_K_estimated_np, 0
        else:
            return -10.0, current_K_estimated_np, 0

    current_potential = calculate_potential(current_K_estimated_np, target_K_np, device)
    progress_reward = 0
    if prev_K_estimated_np is not None:
        prev_potential = calculate_potential(prev_K_estimated_np, target_K_np, device)
        progress_reward = current_potential - prev_potential

    total_reward = progress_reward - 0.05  # Step penalty (kept for compatibility)
    return total_reward, current_K_estimated_np, progress_reward


def estimate_partial_stiffness(grid):
    processed_grid = []
    for row in grid:
        processed_row = []
        for cell in row:
            if isinstance(cell, list):
                processed_row.append(str(cell[0]) if len(cell) == 1 else "type5_0")
            else:
                processed_row.append(str(cell))
        processed_grid.append(processed_row)
    return WFC_library_undirected.perform_FEA(processed_grid)


def build_constraint_mask(grid, fiber_block_library, device):
    grid_height, grid_width = len(grid), len(grid[0])
    num_states = len(fiber_block_library)
    grid_constraints = torch.zeros((1, grid_height, grid_width, num_states), device=device)
    state_keys = list(fiber_block_library.keys())
    for m in range(grid_height):
        for n in range(grid_width):
            for state in grid[m][n]:
                state_index = state_keys.index(state)
                grid_constraints[0, m, n, state_index] = 1.0
    return grid_constraints


# ==================== New two-head action selection ====================
def select_action_two_head(grid, cell_logits, state_probs, fiber_block_library,
                           alpha=0.5, epsilon=0.1, greedy=False):
    """
    Robust two-head selector that handles different tensor shapes:
      - cell_logits: (H,W) or (1,H,W) or (B,H,W)
      - state_probs: (H,W,S) or (1,H,W,S) or (B,H,W,S)
    Returns: (r,c), selected_state_name, combined_logprob (tensor)
    """
    grid_height, grid_width = len(grid), len(grid[0])
    state_keys = list(fiber_block_library.keys())
    S = len(state_keys)

    # helper accessors to support multiple possible shapes
    def get_cell_logit(tensor, r, c):
        if tensor is None:
            return None
        d = tensor.dim()
        if d == 2:  # (H,W)
            return tensor[r, c]
        elif d == 3:  # (1,H,W) or (B,H,W) -> take batch 0
            return tensor[0, r, c]
        elif d == 4:  # (B,1,H,W) or (B,H,W,1) (unlikely) -> try safe indexing
            return tensor[0, 0, r, c] if tensor.size(1) == 1 else tensor[0, r, c, 0]
        else:
            raise ValueError(f"Unsupported cell_logits dim: {d}")

    def get_state_probs(tensor, r, c):
        if tensor is None:
            return None
        d = tensor.dim()
        if d == 3:  # (H,W,S)
            return tensor[r, c, :]
        elif d == 4:  # (1,H,W,S) or (B,H,W,S)
            return tensor[0, r, c, :]
        else:
            raise ValueError(f"Unsupported state_probs dim: {d}")

    # build candidate cells
    cand_cells = [(r, c) for r in range(grid_height) for c in range(grid_width) if len(grid[r][c]) > 1]
    if not cand_cells:
        return None, None, None

    scores = []
    entropies = []
    device_local = state_probs.device if isinstance(state_probs, torch.Tensor) else torch.device('cpu')

    # compute per-candidate score = alpha * (-entropy) + (1-alpha) * cell_logit
    for (r, c) in cand_cells:
        probs_rc = get_state_probs(state_probs, r, c)  # length S
        # mask illegal states
        allowed = grid[r][c]
        allowed_idxs = [state_keys.index(s) for s in allowed]

        mask = torch.full((S,), float('-inf'), device=device_local)
        mask[allowed_idxs] = 0.0

        # convert probs -> logits then apply mask (robust to tiny probs)
        logits_rc = (probs_rc + 1e-9).log() + mask
        probs_rc_masked = F.softmax(logits_rc, dim=-1)
        entropy_rc = -(probs_rc_masked * (probs_rc_masked + 1e-9).log()).sum()
        entropies.append(entropy_rc)

        cell_logit_val = get_cell_logit(cell_logits, r, c)
        score_rc = alpha * (-entropy_rc) + (1 - alpha) * cell_logit_val
        scores.append(score_rc)

    scores_tensor = torch.stack(scores)  # (Ncand,)

    # choose cell
    if greedy or random.random() > epsilon:
        chosen_idx = int(torch.argmax(scores_tensor).item())
        # for numerical stability compute logprob via log_softmax over scores_tensor
        cell_choice_logprob = torch.log_softmax(scores_tensor, dim=0)[chosen_idx]
    else:
        probs_cells = F.softmax(scores_tensor, dim=0)
        chosen_idx = int(torch.multinomial(probs_cells, 1).item())
        cell_choice_logprob = torch.log(probs_cells[chosen_idx] + 1e-9)

    r_chosen, c_chosen = cand_cells[chosen_idx]

    # choose state for picked cell (apply mask)
    probs_selected = get_state_probs(state_probs, r_chosen, c_chosen)
    mask_s = torch.full((S,), float('-inf'), device=device_local)
    idxs = [state_keys.index(s) for s in grid[r_chosen][c_chosen]]
    mask_s[idxs] = 0.0
    logits_s = (probs_selected + 1e-9).log() + mask_s

    if greedy or random.random() > epsilon:
        s_idx = int(torch.argmax(logits_s).item())
        state_choice_logprob = torch.log_softmax(logits_s, dim=-1)[s_idx]
    else:
        probs_s = F.softmax(logits_s, dim=-1)
        s_idx = int(torch.multinomial(probs_s, 1).item())
        state_choice_logprob = torch.log(probs_s[s_idx] + 1e-9)

    selected_state_name = state_keys[s_idx]
    combined_logprob = cell_choice_logprob + state_choice_logprob
    return (r_chosen, c_chosen), selected_state_name, combined_logprob


def propagate_neighbors(grid, i, j, fiber_block_library):
    for (ni, nj) in WFC_library_undirected.get_neighbors(i, j, grid):
        WFC_library_undirected.propagate_constraints(grid, ni, nj, fiber_block_library)


def run_batched_episodes(network, fiber_block_library, goal_K_tensor, initial_grid,
                         num_episodes, max_steps_per_episode, epsilon):
    """
    Runs multiple WFC episodes in parallel to collect training data.
    Uses the two-head network outputs.
    """
    grid_height, grid_width = len(initial_grid), len(initial_grid[0])
    num_states = len(fiber_block_library)

    # --- Initialization for all parallel environments ---
    grids = [copy.deepcopy(initial_grid) for _ in range(num_episodes)]

    # Trajectory data storage for each environment
    batch_log_probs = [[] for _ in range(num_episodes)]
    batch_rewards = [[] for _ in range(num_episodes)]

    # Tracking which environments are still running
    active_envs_mask = [True] * num_episodes

    # Storing previous stiffness for reward calculation
    prev_K_list = [None] * num_episodes

    for step in range(max_steps_per_episode):
        active_indices = [i for i, active in enumerate(active_envs_mask) if active]
        if not active_indices:
            break

        # --- 1. Batch State Preparation ---
        active_grids = [grids[i] for i in active_indices]
        constraint_masks = [build_constraint_mask(g, fiber_block_library, device) for g in active_grids]

        # Stack into a single batch for the network
        batch_constraints = torch.cat(constraint_masks, dim=0)
        batch_goal_K = goal_K_tensor.repeat(len(active_indices), 1)

        # --- 2. Batched Network Inference ---
        cell_logits_batch, state_probs_batch = network(batch_constraints)
        # cell_logits_batch: (B_active, H, W)
        # state_probs_batch: (B_active, H, W, S)

        # --- 3. Parallel Action Selection & Environment Step ---
        next_active_envs_mask = active_envs_mask[:]
        for i, env_idx in enumerate(active_indices):
            grid = grids[env_idx]
            cell_logits = cell_logits_batch[i].unsqueeze(0)  # make batch dim consistent for selector
            state_probs = state_probs_batch[i].unsqueeze(0)

            selected_cell, selected_state, log_prob = select_action_two_head(
                grid, cell_logits, state_probs, fiber_block_library,
                alpha=0.5, epsilon=epsilon, greedy=False
            )

            if selected_cell is None:  # Grid fully collapsed
                next_active_envs_mask[env_idx] = False
                continue

            batch_log_probs[env_idx].append(log_prob)

            try:
                r, c = selected_cell
                grid[r][c] = [selected_state]
                propagate_neighbors(grid, r, c, fiber_block_library)

                if all(len(cell) == 1 for row in grid for cell in row):
                    next_active_envs_mask[env_idx] = False

                current_K = visualize_elastic_module.calculate_stiffness_directions(estimate_partial_stiffness(grid))
                step_reward, _, _ = compute_incremental_reward(
                    current_K, goal_K_tensor.squeeze().cpu().numpy(), prev_K_list[env_idx],
                    is_terminal=False, success=True
                )
                batch_rewards[env_idx].append(step_reward)
                prev_K_list[env_idx] = current_K

            except ConstraintViolation:
                batch_rewards[env_idx].append(-10.0)
                next_active_envs_mask[env_idx] = False

        active_envs_mask = next_active_envs_mask

    # --- 4. Final Reward Calculation for all episodes ---
    final_rewards_list = []
    for i in range(num_episodes):
        is_successful = all(len(cell) == 1 for row in grids[i] for cell in row)

        if is_successful:
            final_K = visualize_elastic_module.calculate_stiffness_directions(estimate_partial_stiffness(grids[i]))
            final_reward_val, _, _ = compute_incremental_reward(
                final_K, goal_K_tensor.squeeze().cpu().numpy(), prev_K_list[i],
                is_terminal=True, success=True
            )
            final_rewards_list.append(final_reward_val)
            if len(batch_rewards[i]) < len(batch_log_probs[i]):
                batch_rewards[i].append(final_reward_val)
        else:
            final_reward_val = -10.0
            final_rewards_list.append(final_reward_val)
            if len(batch_rewards[i]) < len(batch_log_probs[i]):
                batch_rewards[i].append(final_reward_val)

    return batch_log_probs, batch_rewards, final_rewards_list, grids


def greedy_wfc_algorithm_sequential(initial_grid, fiber_block_library, goal_K_tensor):
    grid = copy.deepcopy(initial_grid)
    device = goal_K_tensor.device
    height = len(grid)
    width = len(grid[0])

    print("Greedy Algorithm (Sequential): Starting collapse...")

    for r in range(height):
        for c in range(width):
            if len(grid[r][c]) <= 1:
                continue

            best_state_for_cell = None
            max_potential_for_cell = float('-inf')

            for state in grid[r][c]:
                grid_copy = copy.deepcopy(grid)
                grid_copy[r][c] = [state]

                try:
                    propagate_neighbors(grid_copy, r, c, fiber_block_library)
                    current_K_estimated_np = estimate_partial_stiffness(grid_copy)
                    current_K_estimated_np = visualize_elastic_module.calculate_stiffness_directions(
                        current_K_estimated_np)

                    potential = calculate_potential(
                        k_actual_np=current_K_estimated_np,
                        k_target_np=goal_K_tensor.squeeze().cpu().numpy(),
                        device=device
                    )

                    if potential > max_potential_for_cell:
                        max_potential_for_cell = potential
                        best_state_for_cell = state

                except ConstraintViolation:
                    continue

            if best_state_for_cell is None:
                print(
                    f"Greedy Algorithm: Warning! Could not find any valid prototype at position ({r},{c}). Terminating early.")
                return grid

            grid[r][c] = [best_state_for_cell]

            try:
                propagate_neighbors(grid, r, c, fiber_block_library)
            except ConstraintViolation:
                print(
                    f"Greedy Algorithm: CRITICAL ERROR! Unexpected conflict during propagation after placing at ({r},{c}).")
                return grid

    print("Greedy Algorithm (Sequential): Grid fully collapsed.")
    return grid


def create_trap_scenario(grid_size, fiber_block_library):
    ideal_grid = [[["Type4_90"] for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    target_K_matrix = estimate_partial_stiffness(ideal_grid)
    target_K_vector = visualize_elastic_module.calculate_stiffness_directions(target_K_matrix)
    goal_K_tensor = torch.tensor(target_K_vector).float().unsqueeze(0)
    print("Trap Scenario: Generated a target K requiring strong vertical anisotropy.")

    initial_grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    seed_state = "Type4_0"
    seed_pos = (grid_size[0] // 2, grid_size[1] // 2 - 1)
    initial_grid[seed_pos[0]][seed_pos[1]] = [seed_state]
    print(f"Trap Scenario: Pre-placing a seed block {seed_state} at {seed_pos}.")

    try:
        propagate_neighbors(initial_grid, seed_pos[0], seed_pos[1], fiber_block_library)
    except ConstraintViolation:
        print("Error: Initial scenario setup resulted in a conflict.")
        return None, None

    K_library = WFC_library_undirected.generate_K_library(grid_size, fiber_block_library, num_samples=1)
    goal_K = WFC_library_undirected.generate_random_K(K_library)
    stiffness = visualize_elastic_module.calculate_stiffness_directions(goal_K)
    goal_K_tensor = torch.tensor(stiffness).float().unsqueeze(0)
    initial_grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    return goal_K_tensor, initial_grid


def calculate_batched_policy_loss(batch_log_probs, batch_rewards, gamma=0.9):
    """Calculates the total policy loss for a batch of episodes."""
    total_loss = torch.tensor(0.0, device=device)
    num_trajectories = 0

    for i in range(len(batch_log_probs)):
        log_probs = batch_log_probs[i]
        rewards = batch_rewards[i]

        if not log_probs:
            continue

        num_trajectories += 1
        # Calculate discounted rewards for this single trajectory
        R = 0
        discounted_rewards = []
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, device=device)
        # Normalize rewards for stability
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss for this trajectory
        policy_loss = []
        for log_prob, R_val in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R_val)

        total_loss += torch.stack(policy_loss).sum()

    if num_trajectories == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_trajectories  # Return average loss


# ==================================================================
# Section: Main Execution Function (uses two-head network)
# ==================================================================

def main():
    grid_size = (5, 5)
    fiber_block_library = create_fiber_block_library()
    num_states = len(fiber_block_library)

    # ================== 算法参数 ==================
    random_search_trials = 10000  # <--- 新增：为随机搜索设定评估预算
    num_training_steps = 300
    episodes_per_step = 32
    max_steps_per_episode = grid_size[0] * grid_size[1]
    # ===============================================

    np.set_printoptions(suppress=True, formatter={'float_kind': '{:.4e}'.format})
    print(f"Experiment environment set up, using device: {device}")

    print("\n" + "=" * 25 + " Creating Trap Scenario " + "=" * 25)
    goal_K_tensor, initial_grid_trap = create_trap_scenario(grid_size, fiber_block_library)
    if initial_grid_trap is None:
        print("Scenario creation failed, terminating program.")
        return
    goal_K_tensor = goal_K_tensor.to(device)
    print("=" * 28 + " Trap Scenario Ready " + "=" * 28)

    print("\n" + "=" * 25 + " Running Greedy Algorithm " + "=" * 25)
    greedy_final_grid = greedy_wfc_algorithm_sequential(initial_grid_trap, fiber_block_library, goal_K_tensor)
    greedy_K_vector = visualize_elastic_module.calculate_stiffness_directions(
        estimate_partial_stiffness(greedy_final_grid))
    greedy_final_reward = calculate_potential(greedy_K_vector, goal_K_tensor.squeeze().cpu().numpy(), device)
    print("=" * 28 + " Greedy Algorithm Finished " + "=" * 28)

    # <--- 新增：运行随机搜索算法
    print("\n" + "=" * 23 + " Running Random Search Algorithm " + "=" * 23)
    random_final_grid, random_final_reward = random_wfc_algorithm_monte_carlo(
        initial_grid_trap,
        fiber_block_library,
        goal_K_tensor,
        num_trials=random_search_trials
    )
    if random_final_grid is not None:
        random_K_vector = visualize_elastic_module.calculate_stiffness_directions(
            estimate_partial_stiffness(random_final_grid)
        )
    else:
        # 如果随机搜索没找到任何有效解
        random_K_vector = np.zeros_like(greedy_K_vector)
        random_final_reward = -10.0
    print("=" * 26 + " Random Search Finished " + "=" * 27)

    # <--- 修改：更新最终结果的比较部分
    print("\n\n" + "#" * 30 + " Final Results Comparison " + "#" * 30)
    target_K_vector_np = goal_K_tensor.squeeze().cpu().numpy()

    print("\n--- Target ---")
    print(f"Target Elasticity Matrix (K) : \n{target_K_vector_np}")

    print("\n--- Greedy Algorithm ---")
    print(f"Achieved Elasticity Matrix (K) : \n{greedy_K_vector}")
    print(f"Final Reward (Potential)       : {greedy_final_reward:.6f}")

    print("\n--- Random Search Algorithm ---")
    print(f"Total Trials (Evaluations)     : {random_search_trials}")
    print(f"Achieved Elasticity Matrix (K) : \n{random_K_vector}")
    print(f"Best Reward Found (Potential)  : {random_final_reward:.6f}")

    # <--- 修改：更新最终网格的可视化部分
    print("\n" + "#" * 28 + " Grid Visualization Comparison " + "#" * 28)
    print_grid_to_console(greedy_final_grid, title="Grid Found by Greedy Algorithm")
    if random_final_grid:
        print_grid_to_console(random_final_grid,
                              title=f"Best Grid Found by Random Search (out of {random_search_trials} trials)")

    print("\nExperiment finished.")


if __name__ == "__main__":
    main()
