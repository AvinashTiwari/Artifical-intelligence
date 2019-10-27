# Generating Data for the VAE model

# Importing the libraries

import numpy as np
import random
import os
import gym
from model import make_model

# Setting the Hyperparameters and setting up the folder for the generated data

MAX_FRAMES = 1000
MAX_TRIALS = 200
render_mode = False
DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

# Extracting the data for the VAE

model = make_model(load_model=False)
total_frames = 0
model.make_env(render_mode=render_mode, full_episode=True)
for trial in range(MAX_TRIALS):
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []
    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)
    model.init_random_model_params(stdev=np.random.rand()*0.01)
    model.reset()
    obs = model.env.reset()
    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")
      recording_obs.append(obs)
      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)
      recording_action.append(action)
      obs, reward, done, info = model.env.step(action)
      if done:
        break
    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
