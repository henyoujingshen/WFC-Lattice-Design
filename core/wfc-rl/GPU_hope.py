import time
import torch.optim as optim

import Position_constrain
import WFC_library_undirected # 假设存在
import visualize_elastic_module # 假设存在
from Position_constrain import ConstraintViolation # 假设存在
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
from itertools import combinations
import copy

# ==================================================================
# 代码主体部分
# ==================================================================

# 确认设备 (CPU or GPU)
# 您的设置是正确的，将使用cuda:1（如果存在）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================================================================
# Section: Text Visualization Tool (无需修改)
# ==================================================================
def print_grid_to_console(grid, title="Grid Configuration"):
    """
    Prints the final grid configuration to the console in text format.
    """
    print(f"\n--- {title} ---")
    if not grid or not grid[0]:
        print("[Empty Grid]")
        return

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

    if max_len == 0: max_len = 5

    for r in range(len(grid)):
        row_str = "|"
        for c in range(len(grid[0])):
            cell_content = grid[r][c]
            state = ""
            if isinstance(cell_content, list) and len(cell_content) == 1:
                state = cell_content[0]
            elif isinstance(cell_content, str):
                state = cell_content
            else:
                state = f"({len(cell_content)})" if isinstance(cell_content, list) else "???"
            row_str += f" {state.ljust(max_len)} |"
        print(row_str)
    print("-" * len(row_str))


# ==================================================================
# Section: Neural Network Model (已在GPU上，无需修改)
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

        self.cnn = nn.Sequential(
            nn.Conv2d(num_states, cnn_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels // 2), nn.GELU(),
            nn.Conv2d(cnn_channels // 2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_channels, fused_feature_dim, kernel_size=1),
            nn.BatchNorm2d(fused_feature_dim), nn.GELU()
        )

        self.cell_head = nn.Conv2d(fused_feature_dim, 1, kernel_size=1)
        self.state_head = nn.Conv2d(fused_feature_dim, num_states, kernel_size=1)

    def forward(self, grid_constraints):
        x_grid_spatial = grid_constraints.permute(0, 3, 1, 2).float()
        x_grid_features = self.cnn(x_grid_spatial)
        fused_output = self.fusion_conv(x_grid_features)
        cell_logits = self.cell_head(fused_output).permute(0, 2, 3, 1).squeeze(-1)
        state_logits = self.state_head(fused_output).permute(0, 2, 3, 1)
        state_probs = F.softmax(state_logits, dim=-1)
        return cell_logits, state_probs


# ==================================================================
# Section: Core Algorithm and Helper Functions
# ==================================================================

def create_fiber_block_library():
    prototypes = WFC_library_undirected.create_prototypes()
    return {f"{proto.mesh[:-4]}_{proto.rotation}": proto for proto in prototypes}


def hybrid_reward(current_K, target_K, beta, scale=1e10):
    # 此函数接收的已经是GPU张量，计算也在GPU上完成，无需修改
    current_K, target_K = current_K.float(), target_K.float()
    cosine_sim = F.cosine_similarity(current_K.unsqueeze(0), target_K.unsqueeze(0))
    reward_linear = 2 * (cosine_sim - 0.241) / (1 - 0.241) - 1
    euclidean_dist = torch.norm(current_K - target_K, p=2)
    euclidean_reward = 1.0 - (euclidean_dist / scale)
    return (beta * reward_linear + (1 - beta) * euclidean_reward).item()


def calculate_potential(k_actual_np, k_target_np, device, beta=0.6, L2_scale=1e10):
    # 此函数负责将Numpy数组转为GPU张量，是正确的做法
    k_actual_tensor = torch.from_numpy(np.array(k_actual_np)).float().flatten().to(device)
    k_target_tensor = torch.from_numpy(np.array(k_target_np)).float().flatten().to(device)
    return hybrid_reward(k_actual_tensor, k_target_tensor, beta=beta, scale=L2_scale)


def compute_incremental_reward(current_K_estimated_np, target_K_np, prev_K_estimated_np, is_terminal, success):
    # 此函数是逻辑协调，其调用的 calculate_potential 已经使用了GPU，无需修改
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

    total_reward = progress_reward - 0.05
    return total_reward, current_K_estimated_np, progress_reward


def estimate_partial_stiffness(grid):
    # 此函数涉及字符串处理和Python列表迭代，这是典型的CPU任务，不适合GPU并行化
    processed_grid = []
    for row in grid:
        processed_row = []
        for cell in row:
            if isinstance(cell, list):
                processed_row.append(str(cell[0]) if len(cell) == 1 else "type5_0")
            else:
                processed_row.append(str(cell))
        processed_grid.append(processed_row)
    # 真正的计算瓶颈在于 perform_FEA，您后续会提供
    return WFC_library_undirected.perform_FEA(processed_grid)


def build_constraint_mask(grid, fiber_block_library, device):
    # 此函数在GPU上创建了最终的张量，然后用CPU循环填充它。
    # 这是正确的模式，因为迭代逻辑依赖于CPU上的Python数据结构 (list, dict)。
    grid_height, grid_width = len(grid), len(grid[0])
    num_states = len(fiber_block_library)
    grid_constraints = torch.zeros((1, grid_height, grid_width, num_states), device=device)
    state_keys = list(fiber_block_library.keys())
    for m in range(grid_height):
        for n in range(grid_width):
            # grid[m][n] 是一个字符串列表，这个查找过程在CPU上完成
            for state in grid[m][n]:
                state_index = state_keys.index(state)
                # 将结果写入GPU张量
                grid_constraints[0, m, n, state_index] = 1.0
    return grid_constraints


def select_action_two_head(grid, cell_logits, state_probs, fiber_block_library,
                           alpha=0.5, epsilon=0.1, greedy=False, temperature=1.0):
    # 这个函数接收的已经是GPU张量，并且所有torch运算都在GPU上执行，已经优化得很好。
    grid_height, grid_width = len(grid), len(grid[0])
    state_keys = list(fiber_block_library.keys())
    S = len(state_keys)

    def get_cell_logit(tensor, r, c):
        if tensor is None: return None
        d = tensor.dim()
        if d == 2:
            return tensor[r, c]
        elif d == 3:
            return tensor[0, r, c]
        elif d == 4:
            return tensor[0, 0, r, c] if tensor.size(1) == 1 else tensor[0, r, c, 0]
        else:
            raise ValueError(f"Unsupported cell_logits dim: {d}")

    def get_state_probs(tensor, r, c):
        if tensor is None: return None
        d = tensor.dim()
        if d == 3:
            return tensor[r, c, :]
        elif d == 4:
            return tensor[0, r, c, :]
        else:
            raise ValueError(f"Unsupported state_probs dim: {d}")

    cand_cells = [(r, c) for r in range(grid_height) for c in range(grid_width) if len(grid[r][c]) > 1]
    if not cand_cells:
        return None, None, None

    scores = []
    device_local = state_probs.device

    for (r, c) in cand_cells:
        probs_rc = get_state_probs(state_probs, r, c)
        allowed = grid[r][c]
        allowed_idxs = [state_keys.index(s) for s in allowed]

        mask = torch.full((S,), float('-inf'), device=device_local)
        mask[allowed_idxs] = 0.0

        logits_rc = (probs_rc + 1e-9).log() + mask
        probs_rc_masked = F.softmax(logits_rc, dim=-1)
        entropy_rc = -(probs_rc_masked * (probs_rc_masked + 1e-9).log()).sum()

        cell_logit_val = get_cell_logit(cell_logits, r, c)
        score_rc = alpha * (-entropy_rc) + (1 - alpha) * cell_logit_val
        scores.append(score_rc)

    scores_tensor = torch.stack(scores)

    if greedy or random.random() > epsilon:
        cell_probs = F.softmax(scores_tensor / temperature, dim=0)
        chosen_idx = int(torch.multinomial(cell_probs, 1).item())
        cell_choice_logprob = torch.log(cell_probs[chosen_idx] + 1e-9)
    else:
        probs_cells = F.softmax(scores_tensor, dim=0)
        chosen_idx = random.randrange(len(cand_cells))
        cell_choice_logprob = torch.log(probs_cells[chosen_idx] + 1e-9)

    r_chosen, c_chosen = cand_cells[chosen_idx]

    probs_selected = get_state_probs(state_probs, r_chosen, c_chosen)
    mask_s = torch.full((S,), float('-inf'), device=device_local)
    idxs = [state_keys.index(s) for s in grid[r_chosen][c_chosen]]
    mask_s[idxs] = 0.0
    logits_s = (probs_selected + 1e-9).log() + mask_s

    if greedy or random.random() > epsilon:
        state_probs_temp = F.softmax(logits_s / temperature, dim=-1)
        s_idx = int(torch.multinomial(state_probs_temp, 1).item())
        state_choice_logprob = torch.log_softmax(logits_s, dim=-1)[s_idx]
    else:
        probs_s = F.softmax(logits_s, dim=-1)
        s_idx = random.choice(idxs)
        state_choice_logprob = torch.log(probs_s[s_idx] + 1e-9)

    selected_state_name = state_keys[s_idx]
    combined_logprob = cell_choice_logprob + state_choice_logprob
    return (r_chosen, c_chosen), selected_state_name, combined_logprob


def propagate_neighbors(grid, i, j, fiber_block_library):
    # 这个函数是WFC算法的核心，涉及复杂的逻辑判断和数据结构修改，
    # 是典型的CPU密集型任务，无法直接在GPU上运行。
    for (ni, nj) in WFC_library_undirected.get_neighbors(i, j, grid):
        WFC_library_undirected.propagate_constraints(grid, ni, nj, fiber_block_library)


def run_batched_episodes(network, fiber_block_library, goal_K_tensor, initial_grid,
                         num_episodes, max_steps_per_episode, epsilon):
    # 这个函数中的主循环是按时间步顺序执行的，每个环境的下一步状态依赖于上一步的动作。
    # 这种顺序依赖性使其难以在GPU上完全并行化。
    # 当前的实现（批处理神经网络推理）已经是最高效的模式。
    grid_height, grid_width = len(initial_grid), len(initial_grid[0])
    num_states = len(fiber_block_library)

    grids = [copy.deepcopy(initial_grid) for _ in range(num_episodes)]
    batch_log_probs = [[] for _ in range(num_episodes)]
    batch_rewards = [[] for _ in range(num_episodes)]
    batch_state_probs_history = []
    active_envs_mask = [True] * num_episodes
    prev_K_list = [None] * num_episodes

    for step in range(max_steps_per_episode):
        active_indices = [i for i, active in enumerate(active_envs_mask) if active]
        if not active_indices:
            break

        active_grids = [grids[i] for i in active_indices]
        constraint_masks = [build_constraint_mask(g, fiber_block_library, device) for g in active_grids]
        batch_constraints = torch.cat(constraint_masks, dim=0)

        # 神经网络推理是批处理的，并且在GPU上运行，这是最高效的部分。
        cell_logits_batch, state_probs_batch = network(batch_constraints)

        if state_probs_batch.shape[0] > 0:
            batch_state_probs_history.append(state_probs_batch)

        # 动作选择和环境更新是顺序的，只能在CPU上循环处理
        next_active_envs_mask = active_envs_mask[:]
        for i, env_idx in enumerate(active_indices):
            grid = grids[env_idx]
            cell_logits = cell_logits_batch[i].unsqueeze(0)
            state_probs = state_probs_batch[i].unsqueeze(0)

            selected_cell, selected_state, log_prob = select_action_two_head(
                grid, cell_logits, state_probs, fiber_block_library,
                alpha=0.5, epsilon=epsilon, greedy=False
            )

            if selected_cell is None:
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

            except Position_constrain.ConstraintViolation:
                batch_rewards[env_idx].append(-10.0)
                next_active_envs_mask[env_idx] = False

        active_envs_mask = next_active_envs_mask

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

    return batch_log_probs, batch_rewards, final_rewards_list, grids, batch_state_probs_history


def greedy_wfc_algorithm_sequential(initial_grid, fiber_block_library, goal_K_tensor):
    # 这个算法是“顺序”贪心，其核心逻辑是串行的，无法并行化到GPU。
    grid = copy.deepcopy(initial_grid)
    device = goal_K_tensor.device
    height = len(grid)
    width = len(grid[0])
    print("Greedy Algorithm (Sequential): Starting collapse...")
    for r in range(height):
        for c in range(width):
            if len(grid[r][c]) <= 1: continue
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
                    potential = calculate_potential(k_actual_np=current_K_estimated_np,
                                                    k_target_np=goal_K_tensor.squeeze().cpu().numpy(), device=device)
                    if potential > max_potential_for_cell:
                        max_potential_for_cell = potential
                        best_state_for_cell = state
                except Position_constrain.ConstraintViolation:
                    continue
            if best_state_for_cell is None:
                print(
                    f"Greedy Algorithm: Warning! Could not find any valid prototype at position ({r},{c}). Terminating early.")
                return grid
            grid[r][c] = [best_state_for_cell]
            try:
                propagate_neighbors(grid, r, c, fiber_block_library)
            except Position_constrain.ConstraintViolation:
                print(
                    f"Greedy Algorithm: CRITICAL ERROR! Unexpected conflict during propagation after placing at ({r},{c}).")
                return grid
    print("Greedy Algorithm (Sequential): Grid fully collapsed.")
    return grid


def create_trap_scenario(grid_size, fiber_block_library):
    # 这个函数是CPU上的设置代码，计算量很小，无需修改
    ideal_grid = [[["Type3_0"] for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    target_K_matrix = estimate_partial_stiffness(ideal_grid)
    target_K_vector = visualize_elastic_module.calculate_stiffness_directions(target_K_matrix)
    goal_K_tensor = torch.tensor(target_K_vector).float().unsqueeze(0)
    print("Trap Scenario: Generated a target K requiring strong vertical anisotropy.")

    initial_grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    seed_state = "Type4_0"
    seed_pos = (0, 0)
    initial_grid[seed_pos[0]][seed_pos[1]] = [seed_state]
    print(f"Trap Scenario: Pre-placing a seed block {seed_state} at {seed_pos}.")

    try:
        propagate_neighbors(initial_grid, seed_pos[0], seed_pos[1], fiber_block_library)
    except Position_constrain.ConstraintViolation:
        print("Error: Initial scenario setup resulted in a conflict.")
        return None, None
    return goal_K_tensor, initial_grid


# ==================================================================
# >>>>>>>>>> Section: GPU-Optimized Loss Calculation <<<<<<<<<<
# ==================================================================
def calculate_batched_loss_with_entropy(batch_log_probs, batch_rewards, all_state_probs_history,
                                        entropy_coeff=0.01, gamma=0.9):
    """
    Calculates the total loss for a batch, combining REINFORCE policy loss
    with an entropy bonus.
    **MODIFIED to perform discounted reward calculation directly on the GPU.**
    """
    total_policy_loss = torch.tensor(0.0, device=device)
    num_trajectories = 0

    for i in range(len(batch_log_probs)):
        log_probs_list = batch_log_probs[i]
        rewards = batch_rewards[i]

        if not log_probs_list:
            continue

        num_trajectories += 1

        # --- OPTIMIZATION START ---
        # 原来的代码在这里创建一个Python列表，然后在tensor转换时传输到GPU。
        # 新代码直接在GPU上创建和计算，避免了CPU->GPU的数据传输。

        # 1. 将奖励列表一次性传输到GPU
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

        # 2. 在GPU上创建一个用于存放折扣奖励的空张量
        discounted_rewards = torch.zeros_like(rewards_tensor)

        # 3. 在GPU上反向迭代计算折扣奖励
        R = 0.0
        for t in reversed(range(len(rewards))):
            R = rewards_tensor[t] + gamma * R
            discounted_rewards[t] = R
        # --- OPTIMIZATION END ---

        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # log_probs_list 是一个张量列表，我们将它们堆叠起来
        log_probs_tensor = torch.stack(log_probs_list)

        # 直接用张量进行批量计算，而不是Python循环
        policy_loss = -torch.sum(log_probs_tensor * discounted_rewards)

        total_policy_loss += policy_loss

    if num_trajectories == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    avg_policy_loss = total_policy_loss / num_trajectories

    # --- 熵计算部分已经是在GPU上高效执行，无需修改 ---
    total_entropy = 0.0
    num_batches_for_entropy = 0
    if all_state_probs_history:
        for state_probs_batch in all_state_probs_history:
            log_probs_dist = torch.log(state_probs_batch + 1e-9)
            entropy_per_cell = -torch.sum(state_probs_batch * log_probs_dist, dim=-1)
            total_entropy += entropy_per_cell.mean()
            num_batches_for_entropy += 1

    avg_entropy = total_entropy / num_batches_for_entropy if num_batches_for_entropy > 0 else torch.tensor(0.0,
                                                                                                           device=device)

    final_combined_loss = avg_policy_loss - entropy_coeff * avg_entropy

    return final_combined_loss, avg_policy_loss.item(), avg_entropy.item()


# ==================================================================
# Section: Main Execution Function (无需修改)
# ==================================================================

def main():
    grid_size = (5, 5)
    fiber_block_library = create_fiber_block_library()
    num_states = len(fiber_block_library)

    np.set_printoptions(suppress=True, formatter={'float_kind': '{:.4e}'.format})
    print(f"Experiment environment set up, using device: {device}")

    print("\n" + "=" * 25 + " Creating Trap Scenario " + "=" * 25)
    goal_K_tensor, initial_grid_trap = create_trap_scenario(grid_size, fiber_block_library)
    if initial_grid_trap is None:
        print("Scenario creation failed, terminating program.")
        return
    goal_K_tensor = goal_K_tensor.to(device)
    print("=" * 28 + " Trap Scenario Ready " + "=" * 28)
    target_K_vector_np = goal_K_tensor.squeeze().cpu().numpy()

    print("\n--- Target ---")
    print(f"Target Elasticity Matrix (K) : \n{target_K_vector_np}")

    print("\n" + "=" * 25 + " Running Greedy Algorithm " + "=" * 25)
    greedy_final_grid = greedy_wfc_algorithm_sequential(initial_grid_trap, fiber_block_library, goal_K_tensor)
    greedy_K_vector = visualize_elastic_module.calculate_stiffness_directions(
        estimate_partial_stiffness(greedy_final_grid))
    greedy_final_reward = calculate_potential(greedy_K_vector, goal_K_tensor.squeeze().cpu().numpy(), device)
    print("=" * 28 + " Greedy Algorithm Finished " + "=" * 28)

    print("\n" + "=" * 24 + " Running Batched RL Optimization " + "=" * 25)
    network = WFC_RL_Net_Simplified(grid_size[0], grid_size[1], num_states).to(device)
    optimizer = optim.Adam(network.parameters(), lr=5e-5)

    num_training_steps = 300
    episodes_per_step = 32
    max_steps_per_episode = grid_size[0] * grid_size[1]
    entropy_coeff = 0.01

    best_rl_reward, best_rl_grid = float('-inf'), None

    history_final_loss, history_policy_loss, history_entropy, history_reward = [], [], [], []

    pbar = tqdm(range(num_training_steps), desc="Batched RL Training Progress")
    for step in pbar:
        epsilon = 0.5 * (1 + np.cos(np.pi * step / num_training_steps))
        epsilon = max(epsilon, 0.05)

        batch_log_probs, batch_rewards, final_rewards, final_grids, all_state_probs_history = run_batched_episodes(
            network, fiber_block_library, goal_K_tensor, initial_grid_trap,
            num_episodes=episodes_per_step,
            max_steps_per_episode=max_steps_per_episode,
            epsilon=epsilon
        )

        # 调用优化后的损失函数
        final_loss, policy_loss_val, entropy_val = calculate_batched_loss_with_entropy(
            batch_log_probs, batch_rewards, all_state_probs_history, entropy_coeff=entropy_coeff
        )

        if torch.isfinite(final_loss):
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()

        avg_reward_in_batch = np.mean(final_rewards) if final_rewards else -10.0
        history_final_loss.append(final_loss.item())
        history_policy_loss.append(policy_loss_val)
        history_entropy.append(entropy_val)
        history_reward.append(avg_reward_in_batch)

        pbar.set_postfix({
            'reward': f'{avg_reward_in_batch:.2f}',
            'loss': f'{final_loss.item():.2f}',
            'p_loss': f'{policy_loss_val:.2f}',
            'entropy': f'{entropy_val:.2f}'
        })

        if final_rewards:
            current_best_reward_in_batch = max(final_rewards)
            if current_best_reward_in_batch > best_rl_reward:
                best_rl_reward = current_best_reward_in_batch
                best_grid_idx = final_rewards.index(best_rl_reward)
                best_rl_grid = final_grids[best_grid_idx]
                pbar.write(f"\n🎉 New best solution found! Step {step} | New Best Reward: {best_rl_reward:.4f}")

    print("=" * 27 + " Batched RL Optimization Finished " + "=" * 28)

    print("\n\n" + "#" * 30 + " Training History Log " + "#" * 30)
    print("Epoch,Avg_Reward,Final_Loss,Policy_Loss,Entropy")
    for i in range(len(history_final_loss)):
        print(
            f"{i},{history_reward[i]:.6f},{history_final_loss[i]:.6f},{history_policy_loss[i]:.6f},{history_entropy[i]:.6f}")

    print("\n\n" + "#" * 30 + " Final Results Comparison " + "#" * 30)
    target_K_vector_np = goal_K_tensor.squeeze().cpu().numpy()

    print("\n--- Target ---")
    print(f"Target Elasticity Matrix (K) : \n{target_K_vector_np}")

    print("\n--- Greedy Algorithm ---")
    print(f"Achieved Elasticity Matrix (K) : \n{greedy_K_vector}")
    print(f"Final Reward (Potential)       : {greedy_final_reward:.6f}")

    if best_rl_grid is not None:
        rl_K_vector = visualize_elastic_module.calculate_stiffness_directions(estimate_partial_stiffness(best_rl_grid))
        print("\n--- Reinforcement Learning (RL) ---")
        print(f"Optimal Elasticity Matrix (K)  : \n{rl_K_vector}")
        print(f"Highest Reward (Potential)     : {best_rl_reward:.6f}")
    else:
        print("\n--- Reinforcement Learning (RL) ---")
        print("The RL algorithm failed to find a valid solution.")

    print("\n" + "#" * 28 + " Grid Visualization Comparison " + "#" * 28)
    print_grid_to_console(greedy_final_grid, title="Grid Found by Greedy Algorithm")
    if best_rl_grid is not None:
        print_grid_to_console(best_rl_grid, title="Optimal Grid Found by RL Algorithm")

    print("\nExperiment finished.")


if __name__ == "__main__":
    main()