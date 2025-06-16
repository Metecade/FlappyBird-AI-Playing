import time
from environment import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch


def test_model(model_path, num_episodes=10, render=True):
    env = FlappyBirdEnv(render_mode='human' if render else None)
    agent = DQNAgent(state_dim=4, action_dim=2)

    if not agent.load(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    agent.epsilon = 0.0  # 测试时探索率为0

    scores = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            score = info['score']

            if render:
                env.render()
                time.sleep(0.02)

        scores.append(score)
        print(f"Test Episode {ep + 1}/{num_episodes}, Score: {score}")

    print("\nTest Results:")
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")

    env.close()


if __name__ == '__main__':
    test_model('models/final_model.pth', num_episodes=5, render=True)