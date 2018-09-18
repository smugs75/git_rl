from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np



def distant_render(env, agent, state_size):
    s = env.reset()
    done = False
    img = plt.imshow(env.render(mode='rgb_array')) # only call this once
    while not done:
        img.set_data(env.render(mode='rgb_array')) # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = agent.next_action(np.reshape(s, [1, state_size]))
        s, _, done, _ = env.step(action)
        if done:
            break
