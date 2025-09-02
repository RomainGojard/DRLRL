import pygame
import numpy as np

WHITE = (255,255,255)
RED = (200,50,50)
GREEN = (50,200,50)
BLUE = (50,50,200)
BLACK = (0,0,0)

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
        self.font = pygame.font.SysFont(None, 24)

        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.done = False

    def draw_cube(self, x, y, theta):
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
            self.draw_cube(x, y, theta)

            # Affichage infos
            txt = self.font.render(f"x={x:.2f} y={y:.2f} theta={theta:.2f}", True, BLACK)
            self.screen.blit(txt, (10, 10))

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
                self.state, _, self.done = self.env.step(action)
            else:
                pygame.time.wait(1000)