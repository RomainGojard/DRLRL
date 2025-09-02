from environement_2D import HoverEnv2D
from agent_sarsa_2D import SARSAAgent2D
from game_2D import HoverGame2D as HoverGame

N_X, N_Y, N_VX, N_VY, N_THETA, N_OMEGA = 20, 20, 10, 10, 10, 10
ACTIONS = [0, 1, 2, 3]  # rien, poussée, rot gauche, rot droite

env = HoverEnv2D()
agent = SARSAAgent2D(N_X, N_Y, N_VX, N_VY, N_THETA, N_OMEGA, len(ACTIONS))

N_EPISODES = 4000
for ep in range(N_EPISODES):
    state = env.reset()
    s_idx = env.state_to_indices(*state, N_X, N_Y, N_VX, N_VY, N_THETA, N_OMEGA)
    done = False
    total_r = 0
    while not done:
        a_idx = agent.choose_action(s_idx)
        action = ACTIONS[a_idx]
        next_state, r, done = env.step(action)
        s2_idx = env.state_to_indices(*next_state, N_X, N_Y, N_VX, N_VY, N_THETA, N_OMEGA)
        # SARSA :
        a2_idx = agent.choose_action(s2_idx)
        agent.update(s_idx, a_idx, r, s2_idx, a2_idx, done)
        s_idx = s2_idx
        total_r += r
    if (ep+1) % 500 == 0:
        print(f"Episode {ep+1}, reward={total_r:.1f}, epsilon={agent.epsilon:.2f}")

# lancer le jeu pour visualiser la politique apprise
game = HoverGame(env, agent, ACTIONS)
game.render_loop()