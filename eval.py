from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork
from my_model import QMix
import numpy as np

import torch

def load_network(path, architecture, observation_space, action_space):
    if architecture == QMix:
        model = architecture(np.prod(np.array(observation_space)), 81, action_space)
    else:
        model = architecture(observation_space, action_space)
    
    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )
    return model

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 45
    frames = []

    observation_space = env.observation_space("red_0").shape
    action_space = env.action_space("red_0").n

    q_network_2 = load_network('blue.pt', FinalQNetwork, observation_space, action_space)

    qmix = load_network('qmix_ver1.pt', QMix, observation_space, action_space)

    # random policies
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network_2(observation)
                    # q_values = qmix.get_q_values(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                action = env.action_space(agent).sample()


        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"random.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording random agents")

    # pretrained policies
    frames = []
    env.reset()


    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network_2(observation)
                    # q_values = qmix.get_q_values(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained agents")

    env.close()
