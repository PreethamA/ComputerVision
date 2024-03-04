import numpy as np
import random
from PIL import Image

# Define a simple environment with blurred images
class Environment:
    def __init__(self, image_path, max_blur=5):
        self.image = Image.open(image_path).convert('L')  # Convert to grayscale
        self.max_blur = max_blur
        self.reset()

    def blur_image(self, image, blur_radius):
        blurred_image = np.array(image)
        blurred_image = np.clip(blurred_image + np.random.normal(scale=blur_radius, size=blurred_image.shape), 0, 255)
        return Image.fromarray(blurred_image.astype(np.uint8))

    def reset(self):
        self.blurred_image = self.blur_image(self.image, random.randint(0, self.max_blur))
        return np.array(self.blurred_image)

    def step(self, action):
        # Perform action here if needed
        reward = 0  # Placeholder reward
        done = True  # Single-step environment
        return np.array(self.blurred_image), reward, done

# Q-learning agent
class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.995):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((num_actions, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state, action]
        td_target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (td_target - old_q_value)

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate)

# Main loop
if __name__ == "__main__":
    image_path = "blurred_image.jpeg"  # Path to the blurred JPEG image
    env = Environment(image_path)
    agent = QLearningAgent(num_actions=2)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        print("Episode:", episode, " Total Reward:", total_reward)

