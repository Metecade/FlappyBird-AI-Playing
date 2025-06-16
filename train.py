import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent
import os
import time


def train_dqn(episodes=1000, render=False):
    env = FlappyBirdEnv(render_mode='human' if render else None)
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3)

    # 训练指标
    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    best_score = -np.inf

    # 创建模型保存目录
    if not os.path.exists('models'):
        os.makedirs('models')

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            # 学习
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            step += 1

            if render:
                env.render()

        score = info['score']
        scores.append(score)
        epsilons.append(agent.epsilon)

        # 计算平均得分
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # 保存最佳模型
        if score > best_score:
            best_score = score
            agent.save(f'models/best_model_{ep}_{score}.pth')

        print(f"Episode {ep + 1}/{episodes}, Score: {score}, "
              f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {step}")

    # 保存最终模型
    agent.save('models/final_model.pth')

    # 绘制训练结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Avg Score (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(132)
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(133)
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    train_dqn(episodes=100, render=True)