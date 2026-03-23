import numpy as np
from collections import deque


class PerPromptStatTrackerNFT:
    """Stat tracker with distance-based advantage weighting for GARDO."""
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, dists=None, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        dists = np.array(dists)
        dists = dists / (np.linalg.norm(dists, ord=2, axis=1, keepdims=True) + 1e-8)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        # Store weighted distances for logging
        all_weighted_dists = np.zeros(len(rewards))

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.flatten().tolist())
            self.history_prompts.add(hash(prompt))
        for prompt in unique:
            self.stats[prompt] = np.array(self.stats[prompt])
            prompt_mask = prompts == prompt
            prompt_rewards = rewards[prompt_mask]
            prompt_dists = dists[prompt_mask]
            similarity_matrix = np.dot(prompt_dists, prompt_dists.T).clip(-1, 1)
            distance_matrix = 1 - similarity_matrix
            avg_distances = np.zeros(len(distance_matrix))
            for i in range(len(distance_matrix)):
                mask = np.ones(len(distance_matrix), dtype=bool)
                mask[i] = False
                if mask.sum() > 0:
                    other_distances = distance_matrix[i, mask]
                    avg_distances[i] = np.min(other_distances)
                else:
                    avg_distances[i] = 1.0
            mean = np.mean(self.stats[prompt])
            # Normalize distances: set to 1 for negative advantages, otherwise scale by mean
            weighted_dists = np.where(
                (prompt_rewards - mean) < 0,
                np.ones_like(avg_distances),
                avg_distances / (np.mean(avg_distances) + 1e-8)
            )
            if self.global_std:
                std = np.std(rewards) + 1e-4
            else:
                std = np.std(self.stats[prompt]) + 1e-4
            advantages[prompt_mask] = (prompt_rewards - mean) * weighted_dists
            all_weighted_dists[prompt_mask] = weighted_dists
        return advantages, all_weighted_dists

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0
        assert 0 <= top_percentage <= 100
        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards
            if rewards.size == 0:
                continue
            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue
            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)
            top_rewards = rewards[rewards >= threshold]
            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))
        if not per_prompt_top_means:
            return 0.0
        return np.mean(per_prompt_top_means)


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
