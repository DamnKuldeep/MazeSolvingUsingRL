import random
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import time
from PIL import Image
import cv2
import streamlit as st

st.title("Maze Solver App")
st.write("This app solves a maze using a reinforcement learning agent.")
st.set_option('deprecation.showPyplotGlobalUse', False)

example_input = '''0, 0, 0, 0, 1, 0, 0, 1, 0, 0
0, 1, 0, 1, 1, 0, 0, 1, 1, 0
0, 1, 0, 0, 0, 0, 0, 0, 1, 1
0, 1, 1, 1, 0, 1, 1, 0, 1, 0
0, 1, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 0, 1, 0, 1, 1, 1, 1
0, 1, 0, 1, 0, 0, 0, 0, 0, 0
0, 1, 0, 0, 0, 1, 1, 0, 1, 0
0, 1, 1, 0, 1, 0, 1, 0, 1, 1
0, 0, 0, 0, 1, 0, 0, 0, 0, 0'''

placeholder_text = f"Enter maze as a list of rows separated by commas.\n{example_input}"


maze_list = st.text_area("Enter maze", value="", height=300, placeholder=placeholder_text)

maze = []
for row in maze_list.split("\n"):
    if row.strip():  # skip empty lines
        maze.append(list(map(int, row.strip().split(","))))

maze = np.array(maze)




class MazeEnv():

    def __init__(self, maze):
        super(MazeEnv, self).__init__()

        self.maze = maze
        self.start = (0, 0)
        self.end = (9,9)
        self.current_pos = self.start
        self.agent_pos = self.start
        self.cumulative_reward = 0

        self.action_space = [0,1,2,3]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(10, 10, 2), dtype=np.uint8)

    def reset(self):
        self.current_pos = self.start
        self.agent_pos = self.start
        self.cumulative_reward = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        new_pos = self._get_new_pos(action)
        reward = self._get_reward(new_pos)
        done = self._get_done(new_pos)
        self.cumulative_reward += reward
        if self.maze[new_pos] == 0:
            self.current_pos = new_pos
            self.agent_pos = new_pos
        obs = self._get_obs()
        return obs, reward, done, {'cumulative_reward': self.cumulative_reward}

    def render(self, mode='human'):
        if mode not in ['rgb_array', 'human']:
            raise ValueError(f"Invalid mode '{mode}'. Mode must be either 'rgb_array' or 'human'.")
        if mode == 'rgb_array':
            img = np.zeros((self.maze.shape[0], self.maze.shape[1], 3))
            img[self.maze == 1] = [1, 1, 1]
            img[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 1]
            if self.start == self.agent_pos:
                pass
            else:
                img[self.start[0], self.start[1]] = [255, 0, 0]
            if self.end==self.agent_pos:
                pass
            else:
                img[self.end[0], self.end[1]] = [0, 255, 0]
            img = np.clip(img, 0, 1)
            return img
        elif mode == 'human':
            img = self.render(mode='rgb_array')
            plt.imshow(img)
            plt.show()


    def _get_obs(self):
        obs = np.zeros(self.maze.shape + (2,), dtype=np.uint8)
        obs[self.maze == 1] = [0, 0]
        obs[self.maze == 0] = [255, 255]
        obs[self.start] = [0, 255]
        obs[self.end] = [255, 0]
        obs[self.agent_pos] = [255, 255]
        return obs

    def _get_new_pos(self, action):
        row, col = self.current_pos
        if action == 0:  # move up
            row = max(row-1, 0)  
        elif action == 1:  # move down
            row = min(row+1, self.maze.shape[0]-1)
        elif action == 2:  # move left
            col = max(col-1, 0) 
        elif action == 3:  # move right
            col = min(col+1, self.maze.shape[1]-1)
        return (row, col)

    def _get_reward(self, new_pos):
        if new_pos == self.end:
            return 100
        elif self.maze[new_pos[0], new_pos[1]] == 1:
            return -100
        else:
            return -1

    def _get_done(self, new_pos):
        return new_pos == self.end


class agent:
    def __init__(self, env):
        q_table = np.zeros(
            (env.observation_space.shape[0], env.observation_space.shape[1], len(env.action_space)))
        self.env = env
        alpha = 0.1
        gamma = 0.99
        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.001
        self.q_table = q_table
        self.im = []

        num_episodes = 1000
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            for i in range(200):
                if np.random.uniform() < epsilon:
                    action = random.choice(env.action_space)
                else:
                    action = np.argmax(
                        q_table[env.agent_pos[0], env.agent_pos[1], :])

                b, c = env.agent_pos[0], env.agent_pos[1]
                next_state, reward, done, info = env.step(action)
                if done:
                    break

                old_value = q_table[b, c, action]
                next_max = np.max(
                    q_table[env.agent_pos[0], env.agent_pos[1], :])
                new_value = (1 - alpha) * old_value + alpha * \
                    (reward + gamma * next_max)
                q_table[b, c, action] = new_value

                state = next_state

            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    def solve_maze(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        i = 0   
        self.plots= []     
        while not done:
            prev_pos = self.env.agent_pos
            self.plots.append(plt.gcf())
            self.env.render()
            action = np.argmax(
                self.q_table[self.env.agent_pos[0], self.env.agent_pos[1], :])
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if self.env.agent_pos == prev_pos:
                i += 1
                if i == 10:
                    break
            print(
                f"Taking action {action} in state {(self.env.agent_pos[0], self.env.agent_pos[1])}")
            st.pyplot(clear_figure = True)
            time.sleep(0.5)
            display.clear_output(wait=True)
            
        if done:
            self.plots.append(plt.gcf())
            self.env.render()
            st.pyplot()
            st.write(f"Maze Solved")
        else:
            st.write("The Maze is not solvable")

try:
    uploaded_file = st.file_uploader("Choose a PNG image file", type="png")
    maze_image = Image.open(uploaded_file).convert('L')

    threshold_value = 128
    maze_binary = maze_image.point(lambda x: 255 if x > threshold_value else 0, mode='1')
    
    maze_resized = maze_binary.resize((10, 10), resample=Image.BILINEAR)
    
    maze_array = 1 - np.array(maze_resized, dtype=np.int)
    env = MazeEnv(maze_array)
    
except:
    env= MazeEnv(maze)

def main():

    # Input section
    # st.write("Enter the maze array below (use 0s for free spaces and 1s for walls):")
    # maze_input = st.text_area("Maze array", value="0,0,0,0,0,0,1,0,0,0\n0,1,0,1,1,0,1,0,1,1\n0,1,0,0,1,0,1,0,0,0\n0,0,1,0,1,0,0,1,1,0\n1,0,1,0,1,1,0,0,1,0\n0,0,1,0,0,0,1,0,0,0\n0,1,1,1,1,0,1,1,1,1\n0,0,0,0,1,0,1,0,0,0\n1,1,1,0,1,0,1,0,1,0\n0,0,0,0,1,0,0,0,1,0")
    # maze_array = np.fromstring(maze_input, dtype=int, sep=',')

    # Run button
    if st.button("Solve Maze"):
        # Solve maze
        # env = MazeEnv(maze_array)
        Agent = agent(env)
        st.write("Solving maze...")
        Agent.solve_maze()
        


if __name__ == "__main__":
    main()
