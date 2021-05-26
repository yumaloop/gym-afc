import cv2
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from PIL import Image
from scipy.stats import multivariate_normal
from image_foveation import foveat_img


class AfcEnv(gym.Env):
    """
    Artificial Forvia Control
    action: 
    observation: RGB images ()
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        im_path = "./ADEChallengeData2016/images/training/ADE_train_00019690.jpg"
        an_path = im_path.replace("images", "annotations").replace("jpg", "png")

        im = Image.open(im_path)
        an = Image.open(an_path)
        
        self.im_height = 100
        self.im_width = 100
        
        self.im = np.array(im.resize((self.im_width, self.im_height)))
        self.an = np.array(an)
        
        self.an_height = self.an.shape[0]
        self.an_width = self.an.shape[1]

        self.observation_space = spaces.Box(0, 255, shape=self.im.shape)  # continuous
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([21, 21, 5])
        # self.action_space = spaces.Box(-5, +5, (2,), dtype=np.float32) # continuous
        # self.action_space = spaces.Box(low=-10, high=10, shape=(2,))

        self.WINDOW_SIZE = 600  # 画面サイズの決定
        self.reset()

    def reset(self, init_pos=None):
        # 状態を初期化し、初期の観測値を返す
        self.TIME_STEP = 0
        self.fixs = (
            np.array([int(self.im.shape[0] / 2), int(self.im.shape[1] / 2)])
            if init_pos is None
            else np.array(init_pos)
        )
        observation = foveat_img(self.im, [self.fixs], sigma=0.4)
        return observation

    def step(self, action):
        # actionを実行し、結果を返す
        """
        """
        action = np.squeeze(action)
        print(action)

        self.TIME_STEP += 1
        sd = 0.01 + action[2]
        movement = np.array([int(action[0] * 5), int(action[1] * 5)]) # -5 ~ 5
        self.fixs += movement

        if self.fixs[0] >= self.im.shape[0]:
            self.fixs[0] = self.im.shape[0] - 1
        if self.fixs[0] < 0:
            self.fixs[0] = 0
        if self.fixs[1] >= self.im.shape[1]:
            self.fixs[1] = self.im.shape[1] - 1
        if self.fixs[1] < 0:
            self.fixs[1] = 0

        observation = foveat_img(self.im, [self.fixs], sigma=0.4)
        reward = self.reward(self.fixs, sd, label=23) 

        print("fix:", self.fixs, "reward:", reward, "movement:", movement, "action", action)

        if self.TIME_STEP > 100:
            done = True
        else:
            done = False

        return observation, reward, done, {}

    def render(self, mode="human", close=False):
        # 環境を可視化する
        """
        img = foveat_img(self.im, [self.fixs], sigma=0.25)
        cv2.circle(img, tuple(self.fixs), 3, color=(0, 0, 255), thickness=-1)  # ボールの描画
        cv2.circle(img, tuple(self.fixs), 15, color=(0, 0, 255), thickness=1)  # ボールの描画
        cv2.circle(img, tuple(self.fixs), 30, color=(0, 0, 255), thickness=1)  # ボールの描画
        cv2.imshow("render", img)
        cv2.waitKey(1)
        """
        pass

    def close(self):
        # 環境を閉じて後処理をする
        pass

    def seed(self, seed=None):
        # ランダムシードを固定する
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, sd, label=23):
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
