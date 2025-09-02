import numpy as np
import matplotlib.pyplot as plt
from environement_2D import HoverEnv2D
from agent_sarsa_2D import SARSAAgent2D
from agent_qlearning_2D import QLearningAgent2D

# Actions disponibles
ACTIONS = [0, 1, 2, 3]  # rien, thrust, rot gauche, rot droite

# --------- TRAINING FUNCTION ----------
def train(agent, env_class, episodes=2000, max_steps=600):
    rewards = []

    for ep in range(episodes):
        env = env_class()
        state = env.reset()

        # Discrétiser l'état
        s_idx = env.state_to_indices(*state,
                                     agent.n_x, agent.n_y,
                                     agent.n_vx, agent.n_vy,
                                     agent.n_theta, agent.n_omega)

        # SARSA → choisir la première action
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

        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1}/{episodes} - Reward: {ep_reward:.1f}")

    return rewards


# --------- MAIN COMPARISON ----------
if __name__ == "__main__":
    episodes = 2000

    print("=== Training SARSA ===")
    sarsa_agent = SARSAAgent2D(20, 20, 10, 10, 10, 10, len(ACTIONS))
    sarsa_rewards = train(sarsa_agent, HoverEnv2D, episodes=episodes)

    print("=== Training Q-Learning ===")
    q_agent = QLearningAgent2D(20, 20, 10, 10, 10, 10, len(ACTIONS))
    q_rewards = train(q_agent, HoverEnv2D, episodes=episodes)

    # --------- VISUALISATIONS ----------
    plt.figure(figsize=(14, 6))

    # Courbes lissées des rewards
    def smooth(x, window=50):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.subplot(1, 2, 1)
    plt.plot(smooth(sarsa_rewards), label="SARSA", color="blue")
    plt.plot(smooth(q_rewards), label="Q-Learning", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (moyenne mobile)")
    plt.title("Comparaison des rewards cumulés")
    plt.legend()
    plt.grid(True)

    # Distribution des rewards finaux
    plt.subplot(1, 2, 2)
    plt.hist(sarsa_rewards[-500:], bins=30, alpha=0.7, label="SARSA", color="blue")
    plt.hist(q_rewards[-500:], bins=30, alpha=0.7, label="Q-Learning", color="red")
    plt.xlabel("Reward final (500 derniers épisodes)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des performances finales")
    plt.legend()

    plt.tight_layout()
    plt.show()
