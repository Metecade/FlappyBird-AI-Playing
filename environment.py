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
MAX_SCORE = 100  # 新增：最大得分目标


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
            self.large_font = pygame.font.SysFont(None, 72)  # 新增：用于显示大文本

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
        self.victory = False  # 新增：标记是否达成目标
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
            # 新增：计算管道中心Y坐标
            center_y = top.height + PIPE_GAP / 2
            pipe_dy = center_y - self.bird_y
        else:
            pipe_dx, pipe_dy = SCREEN_WIDTH, 0
            center_y = SCREEN_HEIGHT / 2  # 没有管道时使用屏幕中心

        return np.array([self.bird_y, self.bird_vel_y, pipe_dx, pipe_dy], dtype=np.float32)

    def step(self, action):
        base_reward = 0.1  # 存活奖励
        center_reward = 0  # 新增：中心位置奖励
        victory_reward = 0  # 新增：胜利奖励

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

        # 检查胜利条件
        if self.score >= MAX_SCORE and not self.done:
            self.done = True
            self.victory = True
            victory_reward = 1000  # 达成目标奖励
            print(f"Victory! Achieved {MAX_SCORE} points!")

        # 计算奖励
        if self.done:
            # 如果胜利，给予大奖励；如果失败，给予惩罚
            base_reward = victory_reward if self.victory else -10
        elif self._passed_pipe():
            base_reward = 5  # 成功通过管道奖励
            self.score += 1
        else:
            # 新增：根据小鸟与管道中心的距离给予奖励
            next_pipe = self._get_next_pipe()
            if next_pipe:
                top, bottom = next_pipe
                center_y = top.height + PIPE_GAP / 2
                distance = abs(self.bird_y - center_y)

                # 距离越小，奖励越大（最大奖励为1，当距离为0时）
                # 当距离小于管道间隙的一半时给予正奖励，大于时给予负奖励
                max_distance = PIPE_GAP / 2
                if distance < max_distance:
                    # 在管道间隙内：距离中心越近奖励越高
                    center_reward = 1.0 * (1 - distance / max_distance)
                else:
                    # 在管道间隙外：距离中心越远惩罚越大
                    center_reward = -0.5 * (distance / max_distance - 1)

        # 组合所有奖励
        total_reward = base_reward + center_reward

        self.frame_count += 1
        return self._get_obs(), total_reward, self.done, {
            "score": self.score,
            "victory": self.victory,
            "center_reward": center_reward
        }

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

    # 新增：获取下一个管道
    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe[0].x + PIPE_WIDTH > 50:
                return pipe
        return None

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

            # 新增：绘制管道中心线
            center_y = top_pipe.height + PIPE_GAP / 2
            pygame.draw.line(self.screen, (255, 0, 0),
                             (top_pipe.x, center_y),
                             (top_pipe.x + PIPE_WIDTH, center_y), 2)

        # 绘制鸟
        bird_color = (255, 215, 0) if self.victory else (255, 255, 0)  # 胜利时金色，普通时黄色
        pygame.draw.rect(self.screen, bird_color,
                         (50, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT))

        # 新增：绘制小鸟到中心的连线
        next_pipe = self._get_next_pipe()
        if next_pipe:
            top, bottom = next_pipe
            center_y = top.height + PIPE_GAP / 2
            pygame.draw.line(self.screen, (255, 165, 0),
                             (50 + BIRD_WIDTH // 2, int(self.bird_y + BIRD_HEIGHT // 2)),
                             (top.x + PIPE_WIDTH // 2, int(center_y)), 2)

        # 绘制小鸟此时的速度矢量
        direction = 1 if self.bird_vel_y > 0 else -1
        arrow_length = min(30, abs(self.bird_vel_y) * 5)
        pygame.draw.line(self.screen, (255, 0, 0),
                         (50 + BIRD_WIDTH // 2, int(self.bird_y)),
                         (50 + BIRD_WIDTH // 2, int(self.bird_y) + direction * arrow_length),
                         3)

        # 显示分数
        score_text = self.font.render(f'Score: {self.score}/{MAX_SCORE}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        # 新增：显示胜利信息
        if self.victory:
            victory_text = self.large_font.render('VICTORY!', True, (255, 215, 0))
            text_rect = victory_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            pygame.draw.rect(self.screen, (0, 0, 0), text_rect.inflate(20, 20))
            self.screen.blit(victory_text, text_rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()