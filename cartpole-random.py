import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human") # create a training env with the cartpole problem, set render mode to show a visual window

observation, info = env.reset() # reset env to start a new episode - will get us the first observation along with additional info

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:

    action = env.action_space.sample() # set action to random action

    observation, reward, terminated, truncated, info = env.step(action) # execute the action to update the environment

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            # each action-observation exchange is called a timestamp
    
    total_reward += reward
    episode_over = terminated or truncated

    time.sleep(0.7) # slow down the simulation

print(f"Episode finished. Total reward: {total_reward}")
env.close()