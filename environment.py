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

        self.im = np.array(Image.open(im_path))
        self.an = np.array(Image.open(an_path))

        print("hoge")

        self.observation_space = spaces.Box(0, 255, shape=self.im.shape)  # continuous
        self.action_space = spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([21, 21, 5])
        # self.action_space = spaces.Box(-5, +5, (2,), dtype=np.float32) # continuous
        # self.action_space = spaces.Box(low=-10, high=10, shape=(2,))

        self.WINDOW_SIZE = 600  # 画面サイズの決定
        self.TIME_STEP = 0
        self.reset()

    def reset(self, init_pos=None):
        # 状態を初期化し、初期の観測値を返す
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
        self.TIME_STEP += 1
        sd = 0.1 + action[2] 
        movement = np.array([int((action[0] - 0.5) * 20), int((action[1] - 0.5) * 20)])
        self.fixs += movement

        if self.fixs[0] > self.im.shape[0]:
            self.fixs[0] = self.im.shape[0]
        elif self.fixs[0] < 0:
            self.fixs[0] = 0
        elif self.fixs[1] > self.im.shape[1]:
            self.fixs[1] = self.im.shape[1]
        elif self.fixs[1] < 0:
            self.fixs[1] = 0

        observation = foveat_img(self.im, [self.fixs], sigma=0.4)
        reward = self.reward(self.fixs, sd, label=23)

        if self.TIME_STEP > 100:
            done = True
        else:
            done = False

        return observation, reward, done, {}

    def render(self, mode="human", close=False):
        # 環境を可視化する
        # mg = np.zeros((self.WINDOW_SIZE, self.WINDOW_SIZE, 3)) #画面初期化
        img = foveat_img(self.im, [self.fixs], sigma=0.25)
        cv2.circle(img, tuple(self.fixs), 3, color=(0, 0, 255), thickness=-1)  # ボールの描画
        cv2.circle(img, tuple(self.fixs), 15, color=(0, 0, 255), thickness=1)  # ボールの描画
        cv2.circle(img, tuple(self.fixs), 30, color=(0, 0, 255), thickness=1)  # ボールの描画
        cv2.imshow("render", img)
        cv2.waitKey(1)

    def close(self):
        # 環境を閉じて後処理をする
        pass

    def seed(self, seed=None):
        # ランダムシードを固定する
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, sd, label=23):
        F = multivariate_normal(mean=fixs, cov=[[sd ** 2, 0.0], [0.0, sd ** 2]])
        reward = 0.0
        xx, yy = np.where(self.an == label)
        for (x, y) in zip(xx, yy):
            r = F.pdf([x, y])
            reward += r
        return reward
