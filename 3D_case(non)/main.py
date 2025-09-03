from env import HoverEnv3D
from agent import QLearningAgent3D
from game import visualize_3d  # version PyGame 3D simplifiée

# 🔹 Discrétisation
N_X, N_Y, N_Z = 6, 6, 6
N_VX, N_VY, N_VZ = 5, 5, 5
N_THETA, N_PHI, N_PSI = 6, 6, 6
N_OMEGA_X, N_OMEGA_Y, N_OMEGA_Z = 5, 5, 5

# 🔹 Actions : 
# 0 = rien, 1 = thrust, 2/3 = rot pitch-, pitch+, 4/5 = rot roll-, roll+, 6/7 = rot yaw-, yaw+
ACTIONS = list(range(8))

# Environnement + Agent
env = HoverEnv3D()
agent = QLearningAgent3D(N_X, N_Y, N_Z,
                         N_VX, N_VY, N_VZ,
                         N_THETA, N_PHI, N_PSI,
                         N_OMEGA_X, N_OMEGA_Y, N_OMEGA_Z,
                         len(ACTIONS))

# 🔹 Training
N_EPISODES = 2000
for ep in range(N_EPISODES):
    state = env.reset()
    s_idx = env.state_to_indices(*state,
                                 N_X, N_Y, N_Z,
                                 N_VX, N_VY, N_VZ,
                                 N_THETA, N_PHI, N_PSI,
                                 N_OMEGA_X, N_OMEGA_Y, N_OMEGA_Z)
    done = False
    total_r = 0
    while not done:
        a_idx = agent.choose_action(s_idx)
        action = ACTIONS[a_idx]

        next_state, r, done = env.step(action)
        s2_idx = env.state_to_indices(*next_state,
                                      N_X, N_Y, N_Z,
                                      N_VX, N_VY, N_VZ,
                                      N_THETA, N_PHI, N_PSI,
                                      N_OMEGA_X, N_OMEGA_Y, N_OMEGA_Z)

        agent.update(s_idx, a_idx, r, s2_idx, done)

        s_idx = s2_idx
        total_r += r

    if (ep+1) % 200 == 0:
        print(f"Episode {ep+1}, total reward={total_r:.1f}, epsilon={agent.epsilon:.3f}")

# 🔹 Visualisation (jeu en temps réel)
#game = HoverGame3D(env, agent, ACTIONS)
#game.render_loop()
