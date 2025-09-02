import numpy as np
import pygame
import sys

# =====================
# ENVIRONNEMENT HOVER
# =====================
class HoverEnv:
    def __init__(self,
                 g=0.002,
                 thrust=0.009,
                 pos_min=0.0,
                 pos_max=1.0,
                 vel_min=-0.5,
                 vel_max=0.5,
                 max_steps=200,
                 target_center=0.5,
                 target_halfwidth=0.1):
        self.g = g
        self.thrust = thrust
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.max_steps = max_steps
        self.target_center = target_center
        self.target_halfwidth = target_halfwidth
        self.reset()

    def reset(self):
        self.pos = 0.2 + np.random.uniform(-0.02, 0.02)
        self.vel = 0.0
        self.steps = 0
        return (self.pos, self.vel)

    def step(self, action):
        accel = action * self.thrust - self.g
        self.vel += accel
        self.vel = np.clip(self.vel, self.vel_min, self.vel_max)
        self.pos += self.vel
        self.steps += 1

        done = False
        reward = 0
        if self.pos <= self.pos_min or self.pos >= self.pos_max:
            done = True
            reward = -1.0
        else:
            if abs(self.pos - self.target_center) <= self.target_halfwidth:
                reward = 1.0
            else:
                reward = -0.1

        if self.steps >= self.max_steps:
            done = True
        return (self.pos, self.vel), reward, done

    def state_to_indices(self, pos, vel, n_pos, n_vel):
        pos_norm = (pos - self.pos_min) / (self.pos_max - self.pos_min)
        vel_norm = (vel - self.vel_min) / (self.vel_max - self.vel_min)
        i_pos = int(np.clip(pos_norm * (n_pos - 1), 0, n_pos - 1))
        i_vel = int(np.clip(vel_norm * (n_vel - 1), 0, n_vel - 1))
        return i_pos, i_vel

# =====================
# AGENT Q-LEARNING
# =====================
class QLearningAgent:
    def __init__(self, n_pos, n_vel, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.zeros((n_pos, n_vel, n_actions))

    def choose_action(self, s_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[s_idx[0], s_idx[1], :]))

    def update(self, s_idx, a, r, s2_idx, done):
        q_predict = self.q[s_idx[0], s_idx[1], a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q[s2_idx[0], s2_idx[1], :])
        self.q[s_idx[0], s_idx[1], a] += self.alpha * (q_target - q_predict)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =====================
# ENTRAÎNEMENT
# =====================
N_POS, N_VEL = 30, 30
ACTIONS = [-1, 0, 1]
N_ACTIONS = len(ACTIONS)

env = HoverEnv()
agent = QLearningAgent(N_POS, N_VEL, N_ACTIONS)

N_EPISODES = 2000
for ep in range(N_EPISODES):
    state = env.reset()
    s_idx = env.state_to_indices(state[0], state[1], N_POS, N_VEL)
    total_r = 0
    done = False
    while not done:
        a_idx = agent.choose_action(s_idx)
        action = ACTIONS[a_idx]
        (pos, vel), r, done = env.step(action)
        s2_idx = env.state_to_indices(pos, vel, N_POS, N_VEL)
        agent.update(s_idx, a_idx, r, s2_idx, done)
        s_idx = s2_idx
        total_r += r
    if (ep+1) % 500 == 0:
        print(f"Episode {ep+1}/{N_EPISODES}, reward={total_r:.1f}, epsilon={agent.epsilon:.2f}")

# =====================
# VISUALISATION PYGAME
# =====================
pygame.init()
WIDTH, HEIGHT = 200, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hover Q-learning")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# couleurs
WHITE = (255,255,255)
RED = (200,50,50)
GREEN = (50,200,50)
BLUE = (50,50,200)
BLACK = (0,0,0)

running = True
state = env.reset()
s_idx = env.state_to_indices(state[0], state[1], N_POS, N_VEL)
score = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # action gloutonne
    a_idx = np.argmax(agent.q[s_idx[0], s_idx[1], :])
    action = ACTIONS[a_idx]
    (pos, vel), r, done = env.step(action)
    s_idx = env.state_to_indices(pos, vel, N_POS, N_VEL)
    score += r

    # affichage
    screen.fill(WHITE)

    # bande cible
    y_center = int(env.target_center * HEIGHT)
    y_half = int(env.target_halfwidth * HEIGHT)
    pygame.draw.line(screen, GREEN, (0, y_center-y_half), (WIDTH, y_center-y_half), 2)
    pygame.draw.line(screen, GREEN, (0, y_center+y_half), (WIDTH, y_center+y_half), 2)

    # voiture (rectangle bleu)
    y_pos = int(pos * HEIGHT)
    car_rect = pygame.Rect(WIDTH//2 - 10, y_pos-10, 20, 20)
    pygame.draw.rect(screen, BLUE, car_rect)

    # flèche pour visualiser l'action (thrust)
    if action == 1:
        pygame.draw.polygon(screen, RED, [(WIDTH//2, y_pos-20), (WIDTH//2-5, y_pos-30), (WIDTH//2+5, y_pos-30)])
    elif action == -1:
        pygame.draw.polygon(screen, RED, [(WIDTH//2, y_pos+20), (WIDTH//2-5, y_pos+30), (WIDTH//2+5, y_pos+30)])

    # score
    score_text = font.render(f"Score: {int(score)}", True, BLACK)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
