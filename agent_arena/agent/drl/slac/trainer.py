import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 5,
        num_sequences=8,
        eval_interval=10 ** 4,
        num_eval_episodes=5,
        config=None # halid: added
    ):
        # Env to collect samples.
        self.env = env
        # self.env.seed(seed) ## halid: deleted

        # Env for evaluation.
        self.env_test = env_test
        # self.env_test.seed(2 ** 31 - seed) ## halid: deleted

        # Observations for training and evaluation.
        self.ob = SlacObservation(config.observation_shape, config.action_shape, num_sequences) # halid: modified
        self.ob_test = SlacObservation(config.observation_shape, config.action_shape, num_sequences) # halid: modified

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = config.action_repeat # halid: added
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        self.env.set_train() ## halid: added
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
            t = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            t = self.algo.step(self.env, self.ob, t, False)

            # Update the algorithm.
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
                self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))
                self.env.set_train() ## halid: added

        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, step_env):
        mean_return = 0.0

        for i in range(self.num_eval_episodes):
            self.env_test.set_eval() ## halid: added
            state = self.env_test.reset(episode_id=i) ## halid: modified
            obs = cv2.resize(state['observation']['image'], (64, 64)).transpose(2, 0, 1)  ## halid: modified
            self.ob_test.reset_episode(obs) ## halid: modified
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                # state, reward, done, _ = self.env_test.step(action)
                external_state = self.env_test.step(action) ## halid: modified
                obs, reward, done, = external_state['observation']['image'], external_state['reward'], external_state['done'] ## halid: modified
                obs = cv2.resize(obs, (64, 64)).transpose(2, 0, 1) ## halid: modified
                self.ob_test.append(obs, action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))