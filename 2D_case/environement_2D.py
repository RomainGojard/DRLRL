import numpy as np

class HoverEnv2D:
    def __init__(self,
                 width=2.0, height=2.0,
                 g=0.004, thrust=0.02,
                 max_steps=600,
                 target_pos=(1.4, 0.6), target_radius=0.35):
        self.width = width
        self.height = height
        self.g = g
        self.thrust = thrust
        self.vxmax = 0.1
        self.vymax = 0.1
        self.omegamax = 0.1
        self.max_steps = max_steps
        self.target_pos = np.array(target_pos)
        self.target_radius = target_radius
        self.prev_dist = None
        self.reset()

    def reset(self):
        self.x = 1 + np.random.uniform(-0.05, 0.05)
        self.y = 1 + np.random.uniform(-0.05, 0.05)
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0  # angle en radians
        self.omega = 0.0  # vitesse angulaire
        self.steps = 0
        self.prev_dist = np.linalg.norm(np.array([self.x, self.y]) - self.target_pos)
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
            
        #limitation des vitesses
        self.vx = np.clip(self.vx, -self.vxmax, self.vxmax)
        self.vy = np.clip(self.vy, -self.vymax, self.vymax)
        self.omega = np.clip(self.omega, -self.omegamax, self.omegamax)

        self.x += self.vx
        self.y += self.vy
        self.theta += self.omega
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1

        # Reward
        done = False
        dist = np.linalg.norm(np.array([self.x, self.y]) - self.target_pos)

        # Crash hors zone
        if (self.x < 0 or self.x > self.width or
            self.y < 0 or self.y > self.height):
            done = True
            reward = -50.0

        # Maintien dans la cible
        elif dist < self.target_radius:
            reward = +1.5   # chaque step dans le cercle = bon

        else:
            # Pénalité douce en fonction de la distance
            reward = -0.1 * dist

            # Petit bonus si l’agent se rapproche
            if dist < self.prev_dist:
                reward += 0.2

        self.prev_dist = dist

        # Stabilité (éviter d'être couché sur le côté)
        if abs(self.theta) > np.pi / 4:
            reward -= 0.5

        # Step limit
        if self.steps >= self.max_steps:
            done = True

        return (self.x, self.y, self.vx, self.vy, self.theta, self.omega), reward, done

    def state_to_indices(self, x, y, vx, vy, theta, omega,
                        n_x, n_y, n_vx, n_vy, n_theta, n_omega):
        # Position
        i_x = int(np.clip(x / self.width * (n_x - 1), 0, n_x - 1))
        i_y = int(np.clip(y / self.height * (n_y - 1), 0, n_y - 1))

        # Vitesses normalisées dans [-0.3, 0.3]
        i_vx = int(np.clip((vx + 0.3) / 0.6 * (n_vx - 1), 0, n_vx - 1))
        i_vy = int(np.clip((vy + 0.3) / 0.6 * (n_vy - 1), 0, n_vy - 1))

        # Angle [-pi, pi]
        i_theta = int(np.clip((theta + np.pi) / (2 * np.pi) * (n_theta - 1), 0, n_theta - 1))

        # Vitesse angulaire bornée à [-0.2, 0.2] (exemple)
        i_omega = int(np.clip((omega + 0.2) / 0.4 * (n_omega - 1), 0, n_omega - 1))

        return i_x, i_y, i_vx, i_vy, i_theta, i_omega
