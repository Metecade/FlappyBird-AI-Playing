import pygame
import gym
import numpy as np
import random
from gym import spaces

# 游戏参数
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 640
PIPE_WIDTH = 50
PIPE_GAP = 150
PIPE_INTERVAL = 50
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
GRAVITY = 1.0
JUMP_VELOCITY = -7
PIPE_SPEED = 10
GROUND_HEIGHT = 50

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        self.render_mode = render_mode

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird RL')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        self.action_space = spaces.Discrete(2)  # 0: 不跳, 1: 跳
        low = np.array([0, -10, -SCREEN_WIDTH, -SCREEN_HEIGHT], dtype=np.float32)
        high = np.array([SCREEN_HEIGHT, 10, SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT / 2
        self.bird_vel_y = 0
        self.pipes = []
        self.frame_count = 0
        self.score = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # 找到最近的管道
        next_pipe = None
        for pipe in self.pipes:
            if pipe[0].x + PIPE_WIDTH > 50:
                next_pipe = pipe
                break

        if next_pipe:
            top, bottom = next_pipe
            pipe_dx = top.x - 50
            pipe_dy = (top.height + PIPE_GAP / 2) - self.bird_y
        else:
            pipe_dx, pipe_dy = SCREEN_WIDTH, 0

        return np.array([self.bird_y, self.bird_vel_y, pipe_dx, pipe_dy], dtype=np.float32)

    def step(self, action):
        reward = 0.1  # 存活奖励

        # 执行动作
        if action == 1:
            self.bird_vel_y = JUMP_VELOCITY

        # 应用重力
        self.bird_vel_y += GRAVITY
        self.bird_y += self.bird_vel_y

        # 添加新管道
        if self.frame_count % PIPE_INTERVAL == 0:
            self._add_pipe()

        # 移动管道
        self._move_pipes()
        self._remove_offscreen_pipes()

        # 检查碰撞
        self.done = self._check_collision()

        # 计算奖励
        if self.done:
            reward = -10  # 碰撞惩罚
        elif self._passed_pipe():
            reward = 5  # 成功通过管道奖励
            self.score += 1

        self.frame_count += 1
        return self._get_obs(), reward, self.done, {"score": self.score}

    def _add_pipe(self):
        gap_y = random.randint(100, SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 100)
        top_pipe = pygame.Rect(SCREEN_WIDTH, 0, PIPE_WIDTH, gap_y)
        bottom_pipe = pygame.Rect(SCREEN_WIDTH, gap_y + PIPE_GAP, PIPE_WIDTH,
                                  SCREEN_HEIGHT - (gap_y + PIPE_GAP))
        self.pipes.append((top_pipe, bottom_pipe))

    def _move_pipes(self):
        for i in range(len(self.pipes)):
            top, bottom = self.pipes[i]
            top.x -= PIPE_SPEED
            bottom.x -= PIPE_SPEED

    def _remove_offscreen_pipes(self):
        self.pipes = [pair for pair in self.pipes if pair[0].x + PIPE_WIDTH > 0]

    def _check_collision(self):
        # 创建鸟的碰撞矩形
        bird_rect = pygame.Rect(50, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)

        # 检查上下边界碰撞
        if self.bird_y < 0 or self.bird_y + BIRD_HEIGHT > SCREEN_HEIGHT - GROUND_HEIGHT:
            return True

        # 检查管道碰撞
        for top_pipe, bottom_pipe in self.pipes:
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                return True

        return False

    def _passed_pipe(self):
        for top_pipe, bottom_pipe in self.pipes:
            if top_pipe.x + PIPE_WIDTH == 50:
                return True
        return False

    def render(self, mode='human'):
        if self.render_mode != 'human':
            return

        self.screen.fill((135, 206, 250))  # 天空蓝背景

        # 绘制地面
        pygame.draw.rect(self.screen, (222, 184, 135),
                         (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))

        # 绘制管道
        for top_pipe, bottom_pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 180, 0), top_pipe)
            pygame.draw.rect(self.screen, (0, 180, 0), bottom_pipe)

        # 绘制鸟
        pygame.draw.rect(self.screen, (255, 255, 0),
                         (50, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT))

        # 显示分数
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()