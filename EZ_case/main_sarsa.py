from environment import HoverEnv
from agent_sarsa import SARSAAgent
from game import HoverGame

# paramètres
N_POS, N_VEL = 30, 30
# actions possibles : [0 = rien, 1 = poussée vers le haut]
ACTIONS = [0, 1]

# créer env et agent
env = HoverEnv()
agent = SARSAAgent(N_POS, N_VEL, len(ACTIONS))

# entraînement
N_EPISODES = 2000
for ep in range(N_EPISODES):
    state = env.reset()
    s_idx = env.state_to_indices(state[0], state[1], N_POS, N_VEL)
    done = False
    total_r = 0
    while not done:
        a_idx = agent.choose_action(s_idx)
        action = ACTIONS[a_idx]
        (pos, vel), r, done = env.step(action)
        s2_idx = env.state_to_indices(pos, vel, N_POS, N_VEL)
        a2_idx = agent.choose_action(s2_idx)
        agent.update(s_idx, a_idx, r, s2_idx, a2_idx, done)
        s_idx = s2_idx
        total_r += r
    if (ep+1) % 500 == 0:
        print(f"Episode {ep+1}, reward={total_r:.1f}, epsilon={agent.epsilon:.2f}")

# lancer le jeu pour visualiser la politique apprise
game = HoverGame(env, agent, ACTIONS)
game.render_loop()
