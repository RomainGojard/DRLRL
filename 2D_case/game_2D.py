import pygame
import numpy as np

WHITE = (255,255,255)
RED = (200,50,50)
GREEN = (50,200,50)
BLUE = (50,50,200)
BLACK = (0,0,0)
ORANGE = (255,150,0)

class HoverGame2D:
    def __init__(self, env, agent, actions):
        pygame.init()
        self.env = env
        self.agent = agent
        self.actions = actions

        self.WIDTH, self.HEIGHT = 800, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Hover 2D RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 28)
        self.big_font = pygame.font.SysFont(None, 72, bold=True)

        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0.0  # cumul reward par épisode
        self.last_action = None    # dernière action prise

    def draw_cube(self, x, y, theta, action):
        # Conversion coordonnées logiques -> écran
        px = int(x / self.env.width * self.WIDTH)
        py = int(self.HEIGHT - y / self.env.height * self.HEIGHT)
        size = 30

        # Calcul des coins du carré selon l'angle
        points = []
        for dx, dy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
            rx = dx * size / 2
            ry = dy * size / 2
            # rotation
            rrx = rx * np.cos(theta) - ry * np.sin(theta)
            rry = rx * np.sin(theta) + ry * np.cos(theta)
            points.append((px + rrx, py + rry))
        pygame.draw.polygon(self.screen, BLUE, points)

    def draw_target(self, reached):
        tx, ty = self.env.target_pos
        px = int(tx / self.env.width * self.WIDTH)
        py = int(self.HEIGHT - ty / self.env.height * self.HEIGHT)
        radius = int(self.env.target_radius / self.env.width * self.WIDTH)
        color = GREEN if reached else RED
        pygame.draw.circle(self.screen, color, (px, py), radius)

    def render_loop(self):
        running = True
        self.reset()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(WHITE)
            # Vérifie si la cible est atteinte
            x, y, vx, vy, theta, omega = self.state
            dist = np.linalg.norm(np.array([x, y]) - self.env.target_pos)
            reached = dist < self.env.target_radius

            self.draw_target(reached)
            self.draw_cube(x, y, theta, self.last_action)

            # Texte reward au CENTRE
            reward_text = self.big_font.render(f"{self.episode_reward:.0f}", True, BLACK)
            rect = reward_text.get_rect(center=(self.WIDTH//2, 50))
            self.screen.blit(reward_text, rect)

            pygame.display.flip()
            self.clock.tick(30)

            if not self.done:
                # RL agent joue
                s_idx = self.env.state_to_indices(x, y, vx, vy, theta, omega,
                                                  self.agent.n_x, self.agent.n_y,
                                                  self.agent.n_vx, self.agent.n_vy,
                                                  self.agent.n_theta, self.agent.n_omega)
                a_idx = self.agent.choose_action(s_idx)
                action = self.actions[a_idx]

                self.state, reward, self.done = self.env.step(action)
                self.episode_reward += reward
                self.last_action = action
            else:
                # relancer l'épisode
                self.reset()
        pygame.quit()
