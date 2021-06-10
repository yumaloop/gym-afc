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

        self.im_pil = Image.open(im_path)
        self.an_pil = Image.open(an_path)

        self.im_height = 100
        self.im_width = 100

        self.im = np.array(self.im_pil.resize((self.im_width, self.im_height)))
        self.an = np.array(self.an_pil)

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
        """ """
        action = np.squeeze(action)

        self.TIME_STEP += 1
        sd = 0.01 + action[2]
        movement = np.array([int(action[0] * 5), int(action[1] * 5)])  # -5 ~ 5
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

        print(
            "fix:",
            self.fixs,
            "reward:",
            reward,
            "movement:",
            movement,
            "action",
            action,
        )

        if self.TIME_STEP > 100:
            done = True
        else:
            done = False

        return observation, reward, done, {}

    def render(self, mode="human", close=False):
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
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, sd, label=23):
        if (
            fixs[0] <= 0
            or self.im_height <= fixs[0]
            or fixs[1] < 0
            or self.im_width <= fixs[1]
        ):
            reward = -10
        else:
            reward = 0.001
            a0 = (self.an_height / self.im_height) * fixs[0]
            a1 = (self.an_width / self.im_width) * fixs[1]
            fixs = np.array([a0, a1])
            F = multivariate_normal(
                mean=fixs,
                cov=[
                    [(sd * self.im_height) ** 2, 0.0],
                    [0.0, (sd * self.im_width) ** 2],
                ],
            )
            xx, yy = np.where(self.an == label)
            for (x, y) in zip(xx, yy):
                r = F.pdf([x, y])
                reward += r
        return reward





class AfcEnvPO(gym.Env):
    """
    Artificial Forvia Control (Partially Observable)
    action:
    observation: RGB images ()
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        im_path = "./ADEChallengeData2016/images/training/ADE_train_00010429.jpg"
        an_path = im_path.replace("images", "annotations").replace("jpg", "png")

        self.im_pil = Image.open(im_path)
        self.an_pil = Image.open(an_path)

        self.im = np.array(self.im_pil)
        self.an = np.array(self.an_pil)

        self.im_height = self.im.shape[0]
        self.im_width = self.im.shape[1]
        self.an_height = self.an.shape[0]
        self.an_width = self.an.shape[1]

        self.window_size = 100
        self.observation_shape = (self.window_size, self.window_size, 3)
        self.observation_space = spaces.Box(
            0, 255, shape=self.observation_shape
        )  # continuous
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, init_pos=None):
        self.TIME_STEP = 0
        self.fixs = (
            np.array([int(self.im_height / 2), int(self.im_width / 2)])
            if init_pos is None
            else np.array(init_pos)
        )
        """
        w = int(self.window_size / 2)
        yc = np.random.randint(w, self.im_height - w)
        xc = np.random.randint(w, self.im_width - w)
        self.fixs = np.array([yc, xc])
        """
        observation = self.get_observation(self.fixs)
        return observation

    def step(self, action):
        self.TIME_STEP += 1
        """ 
        action_map = {
            0: [0, -1],
            1: [0,  1],
            2: [0, -1],
            3: [0,  1],
            4: [1, -1],
            5: [1,  1],
            6: [1, -1],
            7: [1,  1],
        }
        """
        movement = np.array([int(action[0] * 2), int(action[1] * 2)])  # -2 ~ 2
        self.fixs += movement
        print(action, movement, self.fixs)

        w = int(self.window_size / 2)
        yc, xc = self.fixs[0], self.fixs[1]
        ymax, xmax = self.im.shape[0], self.im.shape[1]

        if yc >= ymax - w:
            self.fixs[0] = ymax - w - 1
        if yc < w:
            self.fixs[0] = w
        if xc >= xmax - w:
            self.fixs[1] = xmax - w - 1
        if xc < w:
            self.fixs[1] = w

        observation = self.get_observation(self.fixs)
        reward = self.reward(self.fixs, label=20)  # label=20: chair
        # print("fix:", self.fixs, "reward:", reward, "movement:", movement, "action", action)

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
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, label=23):
        if (
            fixs[0] <= 0
            or self.im_height <= fixs[0]
            or fixs[1] < 0
            or self.im_width <= fixs[1]
        ):
            reward = -100
        else:
            reward = 0.0
            yc, xc = fixs[0], fixs[1]
            w = int(self.window_size / 2)
            box = (xc - w, yc - w, xc + w, yc + w)  # (left, upper, right, lower)
            an = np.array(self.an_pil.crop(box))
            xx, yy = np.where(an == label)
            reward_ = len(xx) / (self.window_size ** 2)
            reward = 100 * reward_
        return reward

    def get_observation(self, fixs):
        yc, xc = fixs[0], fixs[1]
        w = int(self.window_size / 2)
        box = (xc - w, yc - w, xc + w, yc + w)  # (left, upper, right, lower)
        img = np.array(self.im_pil.crop(box))
        observation = img
        # observation = foveat_img(self.im, [self.fixs], sigma=0.4)
        return observation



class AfcEnvGrid(gym.Env):
    """
    Artificial Forvia Control (Partially Observable)
    action:
    observation: RGB images ()
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        im_path = "./ADEChallengeData2016/images/training/ADE_train_00010429.jpg"
        an_path = im_path.replace("images", "annotations").replace("jpg", "png")

        self.im_pil = Image.open(im_path)
        self.an_pil = Image.open(an_path)

        self.im = np.array(self.im_pil)
        self.an = np.array(self.an_pil)

        self.im_height = self.im.shape[0]
        self.im_width = self.im.shape[1]
        self.an_height = self.an.shape[0]
        self.an_width = self.an.shape[1]

        self.window_size = 100
        self.observation_shape = (self.window_size, self.window_size, 3)
        self.observation_space = spaces.Box(
            0, 255, shape=self.observation_shape
        )  # continuous
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, init_pos=None):
        self.TIME_STEP = 0
        self.fixs = (
            np.array([int(self.im_height / 2), int(self.im_width / 2)])
            if init_pos is None
            else np.array(init_pos)
        )
        """
        w = int(self.window_size / 2)
        yc = np.random.randint(w, self.im_height - w)
        xc = np.random.randint(w, self.im_width - w)
        self.fixs = np.array([yc, xc])
        """
        observation = self.get_observation(self.fixs)
        return observation

    def step(self, action):
        self.TIME_STEP += 1
        """ 
        action_map = {
            0: [0, -1],
            1: [0,  1],
            2: [0, -1],
            3: [0,  1],
            4: [1, -1],
            5: [1,  1],
            6: [1, -1],
            7: [1,  1],
        }
        """
        movement = np.array([int(action[0] * 2), int(action[1] * 2)])  # -2 ~ 2
        self.fixs += movement
        print(action, movement, self.fixs)

        w = int(self.window_size / 2)
        yc, xc = self.fixs[0], self.fixs[1]
        ymax, xmax = self.im.shape[0], self.im.shape[1]

        if yc >= ymax - w:
            self.fixs[0] = ymax - w - 1
        if yc < w:
            self.fixs[0] = w
        if xc >= xmax - w:
            self.fixs[1] = xmax - w - 1
        if xc < w:
            self.fixs[1] = w

        observation = self.get_observation(self.fixs)
        reward = self.reward(self.fixs, label=20)  # label=20: chair
        # print("fix:", self.fixs, "reward:", reward, "movement:", movement, "action", action)

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
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, fixs, label=23):
        if (
            fixs[0] <= 0
            or self.im_height <= fixs[0]
            or fixs[1] < 0
            or self.im_width <= fixs[1]
        ):
            reward = -100
        else:
            reward = 0.0
            yc, xc = fixs[0], fixs[1]
            w = int(self.window_size / 2)
            box = (xc - w, yc - w, xc + w, yc + w)  # (left, upper, right, lower)
            an = np.array(self.an_pil.crop(box))
            xx, yy = np.where(an == label)
            reward_ = len(xx) / (self.window_size ** 2)
            reward = 100 * reward_
        return reward

    def get_observation(self, fixs):
        yc, xc = fixs[0], fixs[1]
        w = int(self.window_size / 2)
        box = (xc - w, yc - w, xc + w, yc + w)  # (left, upper, right, lower)
        img = np.array(self.im_pil.crop(box))
        observation = img
        # observation = foveat_img(self.im, [self.fixs], sigma=0.4)
        return observation
