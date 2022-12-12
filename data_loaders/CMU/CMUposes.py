import pickle as pkl
import numpy as np
import os
from .dataset import Dataset

class CMUPoses(Dataset):
    dataname = "CMU"

    def __init__(self, datapath="/home/tom/motion-diffusion-model/dataset/CMUPoses", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "CMUPoses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._time = [x[:,0] for x in data]
        self._imu = [x[:,1:9*14+1] for x in data]
        self._pose = [x[:,9*14+1:9*14+25+1] for x in data]
        self._root = [x[:,-3:] for x in data]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        # self._joints = [x for x in data["joints3D"]]

        # self._actions = [x for x in data["y"]]

        # total_num_actions = 12
        # self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        # keep_actions = np.arange(0, total_num_actions)
        #
        self._action_to_label = 0
        # self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        #
        # self._action_classes = humanact12_coarse_action_enumerator

    # def _load_joints3D(self, ind, frame_ix):
    #     return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose

    def _load_pose(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix]
        return pose

    def _load_imu(self, ind, frame_ix):
        imu = self._imu[ind][frame_ix].reshape(-1, 14, 9)
        return imu

    def _load_root(self, ind, frame_ix):
        imu = self._root[ind][frame_ix]
        return imu