import numpy as np

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
