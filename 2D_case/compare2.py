import numpy as np
import matplotlib.pyplot as plt
from environement_2D import HoverEnv2D
from agent_sarsa_2D import SARSAAgent2D
from agent_qlearning_2D import QLearningAgent2D

ACTIONS = [0, 1, 2, 3]  # rien, thrust, rot gauche, rot droite


# --------- TRAINING FUNCTION ----------
def train(agent, env_class, episodes=2000, max_steps=600):
    rewards = []
    final_positions = []

    for ep in range(episodes):
        env = env_class()
        state = env.reset()

        s_idx = env.state_to_indices(*state,
                                     agent.n_x, agent.n_y,
                                     agent.n_vx, agent.n_vy,
                                     agent.n_theta, agent.n_omega)

        if isinstance(agent, SARSAAgent2D):
            a = agent.choose_action(s_idx)

        ep_reward = 0
        for _ in range(max_steps):
            if isinstance(agent, SARSAAgent2D):
                action = ACTIONS[a]
                new_state, r, done = env.step(action)
                s2_idx = env.state_to_indices(*new_state,
                                              agent.n_x, agent.n_y,
                                              agent.n_vx, agent.n_vy,
                                              agent.n_theta, agent.n_omega)
                a2 = agent.choose_action(s2_idx)
                agent.update(s_idx, a, r, s2_idx, a2, done)
                s_idx, a = s2_idx, a2

            else:  # Q-Learning
                a = agent.choose_action(s_idx)
                action = ACTIONS[a]
                new_state, r, done = env.step(action)
                s2_idx = env.state_to_indices(*new_state,
                                              agent.n_x, agent.n_y,
                                              agent.n_vx, agent.n_vy,
                                              agent.n_theta, agent.n_omega)
                agent.update(s_idx, a, r, s2_idx, done)
                s_idx = s2_idx

            ep_reward += r
            
            if done:
                break

        rewards.append(ep_reward)
        final_positions.append(new_state[:2])  # on garde juste (x, y)

    return rewards, np.array(final_positions)


# --------- MULTI-RUNS AVERAGING ----------
def multi_run(agent_class, env_class, runs=5, episodes=2000):
    all_rewards = []
    all_positions = []

    for r in range(runs):
        print(f"Run {r+1}/{runs} with {agent_class.__name__}")
        agent = agent_class(20, 20, 10, 10, 10, 10, len(ACTIONS))
        rewards, positions = train(agent, env_class, episodes=episodes)
        all_rewards.append(rewards)
        all_positions.append(positions)

    avg_rewards = np.mean(all_rewards, axis=0)
    return avg_rewards, np.vstack(all_positions)


# --------- MAIN ----------
if __name__ == "__main__":
    episodes = 2000
    runs = 5

    print("=== Training SARSA (multi-runs) ===")
    sarsa_rewards, sarsa_positions = multi_run(SARSAAgent2D, HoverEnv2D, runs=runs, episodes=episodes)

    print("=== Training Q-Learning (multi-runs) ===")
    q_rewards, q_positions = multi_run(QLearningAgent2D, HoverEnv2D, runs=runs, episodes=episodes)

    # --------- VISUALISATIONS ----------
    plt.figure(figsize=(15, 6))

    # Courbes lissées
    def smooth(x, window=50):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.subplot(1, 3, 1)
    plt.plot(smooth(sarsa_rewards), label="SARSA", color="blue")
    plt.plot(smooth(q_rewards), label="Q-Learning", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (moyenne mobile)")
    plt.title("Comparaison des rewards moyens")
    plt.legend()
    plt.grid(True)

    # Normaliser les positions dans [0, 1]
    sarsa_x = sarsa_positions[:, 0] / HoverEnv2D().width
    sarsa_y = sarsa_positions[:, 1] / HoverEnv2D().height

    q_x = q_positions[:, 0] / HoverEnv2D().width
    q_y = q_positions[:, 1] / HoverEnv2D().height

    # Heatmap SARSA
    plt.subplot(1, 3, 2)
    heatmap_sarsa, xedges, yedges = np.histogram2d(
        sarsa_x, sarsa_y, bins=30, range=[[0, 1], [0, 1]]
    )
    plt.imshow(
        heatmap_sarsa.T,
        origin="lower",
        cmap="Blues",
        extent=[0, 1, 0, 1],
        aspect="auto"
    )
    plt.colorbar(label="Fréquence")
    plt.title("Positions finales - SARSA")
    plt.xlabel("x (normalisé)")
    plt.ylabel("y (normalisé)")

    # Heatmap Q-Learning
    plt.subplot(1, 3, 3)
    heatmap_q, xedges, yedges = np.histogram2d(
        q_x, q_y, bins=30, range=[[0, 1], [0, 1]]
    )
    plt.imshow(
        heatmap_q.T,
        origin="lower",
        cmap="Reds",
        extent=[0, 1, 0, 1],
        aspect="auto"
    )
    plt.colorbar(label="Fréquence")
    plt.title("Positions finales - Q-Learning")
    plt.xlabel("x (normalisé)")
    plt.ylabel("y (normalisé)")


    plt.tight_layout()
    plt.show()
