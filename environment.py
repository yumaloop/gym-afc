import cv2
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from PIL import Image
from scipy.stats import multivariate_normal
from image_foveation import foveat_img


class AfcEnvGrid(gym.Env):
    """
    Artificial Forvia Control
    action: 
    observation: RGB images ()
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.im_path = "./ADEChallengeData2016/images/training/ADE_train_00010429.jpg"
        self.an_path = self.im_path.replace("images", "annotations").replace("jpg", "png")

        self.im_pil = Image.open(self.im_path)
        self.an_pil = Image.open(self.an_path)
        
        self.im_height = 100
        self.im_width = 100
        
        self.im = np.array(self.im_pil.resize((self.im_width, self.im_height)))
        self.an = np.array(self.an_pil)
        
        self.an_height = self.an.shape[0]
        self.an_width = self.an.shape[1]

        # State: 2D-Grid-Space (100 x 100)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.im_height), spaces.Discrete(self.im_width)))
        # Action: 8 types
        self.action_space = spaces.Discrete(8)
 
        self.reset()

    def reset(self, init_pos=None):
        # 状態を初期化し、初期の観測値を返す
        self.TIME_STEP = 0
        self.TIME_STEP_LEN = 100
        self.fixs = np.array([int(self.im_height/2), int(self.im_width/2)])
        observation = self.fixs
        return observation

    def step(self, action):
        self.TIME_STEP += 1

        # change fixation point
        if action == 0:
            movement = np.array([0, -1]) # left
        elif action == 1:
            movement = np.array([0,  1]) # right
        elif action == 2:
            movement = np.array([1, -1]) # up-left
        elif action == 3:
            movement = np.array([1,  0]) # up
        elif action == 4:
            movement = np.array([1,  1]) # up-right
        elif action == 5:
            movement = np.array([-1,-1]) # down-left
        elif action == 6:
            movement = np.array([-1, 0]) # down
        elif action == 7:
            movement = np.array([-1, 1]) # down-right
        self.fixs += movement
        

        # boundary constraint
        if self.fixs[0] >= self.im_height:
            self.fixs[0] = self.im_height - 1
        if self.fixs[0] < 0:
            self.fixs[0] = 0
        if self.fixs[1] >= self.im_width:
            self.fixs[1] = self.im_width - 1
        if self.fixs[1] < 0:
            self.fixs[1] = 0
        observation = self.fixs
        reward = self.reward(self.fixs) 

        # print("fix:", self.fixs, "reward:", reward, "movement:", movement, "action", action)

        if self.TIME_STEP > self.TIME_STEP_LEN:
            done = True
        else:
            done = False

        return observation, reward, done, {}

    def render(self, mode="human", close=False):
        # 環境を可視化する
        pass

    def close(self):
        # 環境を閉じて後処理をする
        pass

    def seed(self, seed=None):
        # ランダムシードを固定する
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, label=20, window_size=50):
        if fixs[0] <= 0 or self.im_height <= fixs[0] or fixs[1] < 0 or self.im_width <= fixs[1]:
            reward = -1
        else:
            # label: 20 (chair)
            yc = int(fixs[0] * (self.an_height / self.im_height))
            xc = int(fixs[1] * (self.an_width / self.im_width))
            w = int(window_size / 2)
            box = (xc - w, yc - w, xc + w, yc + w)  # (left, upper, right, lower)
            an_part = np.array(self.an_pil.crop(box))
            xx, yy = np.where(an_part == label)
            reward = len(xx) / (window_size ** 2) # 0 ~ 1
        return float(reward)

        """
        if fixs[0] <= 0 or self.im_height <= fixs[0] or fixs[1] < 0 or self.im_width <= fixs[1]:
            reward = -10
        else:
            reward = 0.001
            a0 = (self.an_height/self.im_height) * fixs[0]
            a1 = (self.an_width/self.im_width) * fixs[1]
            fixs = np.array([a0, a1])
            F = multivariate_normal(mean=fixs, cov=[[(sd * self.im_height) ** 2, 0.0], [0.0, (sd * self.im_width) ** 2]])
            xx, yy = np.where(self.an == label)
            for (x, y) in zip(xx, yy):
                r = F.pdf([x, y])
                reward += r
        return reward
        """

