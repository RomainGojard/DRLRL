import numpy as np

class HoverEnv2D:
    def __init__(self,
                 width=2.0, height=2.0,
                 g=0.004, thrust=0.02,
                 max_steps=600,
                 target_pos=(1.6, 0.4), target_radius=0.35):
        self.width = width
        self.height = height
        self.g = g
        self.thrust = thrust
        self.max_steps = max_steps
        self.target_pos = np.array(target_pos)
        self.target_radius = target_radius
        self.reset()

    def reset(self):
        self.x = 0.2 + np.random.uniform(-0.05, 0.05)
        self.y = 1.7 + np.random.uniform(-0.05, 0.05)
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0  # angle en radians
        self.omega = 0.0  # vitesse angulaire
        self.steps = 0
        return (self.x, self.y, self.vx, self.vy, self.theta, self.omega)

    def step(self, action):
        # Actions: 0 = rien, 1 = poussée, 2 = rot gauche, 3 = rot droite
        if action == 1:
            fx = self.thrust * np.sin(self.theta)
            fy = self.thrust * np.cos(self.theta)
            self.vx += fx
            self.vy += fy - self.g
        else:
            self.vy -= self.g

        if action == 2:
            self.omega -= 0.05
        elif action == 3:
            self.omega += 0.05

        self.x += self.vx
        self.y += self.vy
        self.theta += self.omega
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1

        done = False
        dist = np.linalg.norm(np.array([self.x, self.y]) - self.target_pos)
        if (self.x < 0 or self.x > self.width or
            self.y < 0 or self.y > self.height):
            done = True
            reward = -10
        elif dist < self.target_radius:
            reward = 1.0  # Récompense à chaque pas dans le cercle
        else:
            reward = -1.0  # pénalité hors du cercle

        if self.steps >= self.max_steps:
            done = True

        return (self.x, self.y, self.vx, self.vy, self.theta, self.omega), reward, done

    def state_to_indices(self, x, y, vx, vy, theta, omega,
                         n_x, n_y, n_vx, n_vy, n_theta, n_omega):
        i_x = int(np.clip(x / self.width * (n_x - 1), 0, n_x - 1))
        i_y = int(np.clip(y / self.height * (n_y - 1), 0, n_y - 1))
        i_vx = int(np.clip((vx + 1) / 2 * (n_vx - 1), 0, n_vx - 1))
        i_vy = int(np.clip((vy + 1) / 2 * (n_vy - 1), 0, n_vy - 1))
        i_theta = int(np.clip((theta + np.pi) / (2 * np.pi) * (n_theta - 1), 0, n_theta - 1))
        i_omega = int(np.clip((omega + 1) / 2 * (n_omega - 1), 0, n_omega - 1))
        return i_x, i_y, i_vx, i_vy, i_theta, i_omega