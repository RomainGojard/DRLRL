import numpy as np

class HoverEnv3D:
    def __init__(self,
                 width=2.0, height=2.0, depth=2.0,
                 g=0.004, thrust=0.02,
                 max_steps=600,
                 target_pos=(1.0, 1.0, 1.0), target_radius=0.35):
        self.width = width
        self.height = height
        self.depth = depth
        self.g = g
        self.thrust = thrust
        self.max_steps = max_steps
        self.target_pos = np.array(target_pos)
        self.target_radius = target_radius
        self.reset()

    def reset(self):
        self.x = 1 + np.random.uniform(-0.05, 0.05)
        self.y = 1 + np.random.uniform(-0.05, 0.05)
        self.z = 1 + np.random.uniform(-0.05, 0.05)
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        self.omega_roll, self.omega_pitch, self.omega_yaw = 0.0, 0.0, 0.0
        self.steps = 0
        self.prev_dist = np.linalg.norm([self.x, self.y, self.z] - self.target_pos)
        return (self.x, self.y, self.z,
                self.vx, self.vy, self.vz,
                self.roll, self.pitch, self.yaw,
                self.omega_roll, self.omega_pitch, self.omega_yaw)

    def step(self, action):
        # Actions:
        # 0 = rien
        # 1 = poussée
        # 2 = roll gauche, 3 = roll droite
        # 4 = pitch bas, 5 = pitch haut
        # 6 = yaw gauche, 7 = yaw droite

        if action == 1:  # poussée suivant orientation
            # direction de poussée en fonction des angles (simplifié)
            fx = self.thrust * np.sin(self.yaw) * np.cos(self.pitch)
            fy = self.thrust * np.cos(self.yaw) * np.cos(self.pitch)
            fz = self.thrust * np.cos(self.roll)
            self.vx += fx
            self.vy += fy
            self.vz += fz - self.g
        else:
            self.vz -= self.g

        # Rotation
        if action == 2: self.omega_roll -= 0.05
        elif action == 3: self.omega_roll += 0.05
        elif action == 4: self.omega_pitch -= 0.05
        elif action == 5: self.omega_pitch += 0.05
        elif action == 6: self.omega_yaw -= 0.05
        elif action == 7: self.omega_yaw += 0.05

        # Update état
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        self.roll += self.omega_roll
        self.pitch += self.omega_pitch
        self.yaw += self.omega_yaw
        self.steps += 1

        # Reward
        done = False
        pos = np.array([self.x, self.y, self.z])
        dist = np.linalg.norm(pos - self.target_pos)

        if (self.x < 0 or self.x > self.width or
            self.y < 0 or self.y > self.height or
            self.z < 0 or self.z > self.depth):
            done = True
            reward = -50.0
        elif dist < self.target_radius:
            reward = +2.0
        else:
            reward = -0.1 * dist
            if dist < self.prev_dist:
                reward += 0.2

        self.prev_dist = dist

        # stabilité
        if abs(self.roll) > np.pi/4 or abs(self.pitch) > np.pi/4:
            reward -= 0.5

        if self.steps >= self.max_steps:
            done = True

        return (self.x, self.y, self.z,
                self.vx, self.vy, self.vz,
                self.roll, self.pitch, self.yaw,
                self.omega_roll, self.omega_pitch, self.omega_yaw), reward, done
