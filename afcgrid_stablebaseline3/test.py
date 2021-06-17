import cv2
import gym
import numpy as np
from stable_baselines3 import DQN
from environment import AfcEnvGrid

model = DQN.load("dqn")
env = AfcEnvGrid()

img = np.uint8(cv2.imread("./data/images/ADE_train_00010429.jpg"))
frame_rate = 24.0 # fps
size = (img.shape[1], img.shape[0]) # image size
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('./video/out.mp4', fmt, frame_rate, size) # ライター作成

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    print(env.fixs, reward)
    
    # img = np.zeros((env.im_height, env.im_width, 3), dtype=np.uint8) #画面初期化
    scale  = np.array([int(env.an_height/env.im_height), int(env.an_width/env.im_width)])
    fixs = env.fixs * scale
    fixs = [fixs[1], fixs[0]]
    print(fixs)

    img = np.uint8(cv2.imread("./data/images/ADE_train_00010429.jpg"))
    cv2.circle(img, tuple(fixs), 1, (0, 255, 0), thickness=1) #ゴールの描画
    cv2.circle(img, tuple(fixs), 15, (0, 255, 0), thickness=1) #ゴールの描画
    writer.write(img)

    # env.render()
    if done:
      obs = env.reset()

env.close()
writer.release()
