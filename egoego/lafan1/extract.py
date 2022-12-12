import re, os, ntpath
import numpy as np
import json 
import cv2 
from . import utils

import torch 

import pytorch3d.transforms as transforms 

from scipy.spatial.transform import Rotation as R

channelmap = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}

channelmap_inv = {
    "x": "Xrotation",
    "y": "Yrotation",
    "z": "Zrotation",
}

ordermap = {
    "x": 0,
    "y": 1,
    "z": 2,
}


class Anim(object):
    """
    A very basic animation object
    """

    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line:
            continue
        if "MOTION" in line:
            continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "{" in line:
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis : 2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(" ")
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(bvh_path, actors, window=50, offset=20, train=True, stats=False, datset='LAFAN'):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = sorted(os.listdir(bvh_path))

    for file in bvh_files:
        if file.endswith(".bvh"):
            file_info = ntpath.basename(file[:-4]).split("_")
            seq_name = file_info[0]
            subject = file_info[1]

            if (not train) and (file_info[-1] == "LRflip"):
                continue

            if stats and (file_info[-1] == "LRflip"):
                continue

            # seq_name, subject = ntpath.basename(file[:-4]).split("_")
            if subject in actors:
                print("Processing file {}".format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                while i + window < anim.pos.shape[0]:
                    q, x = utils.quat_fk(
                        anim.quats[i : i + window],
                        anim.pos[i : i + window],
                        anim.parents,
                    )
                    # Extract contacts
                    c_l, c_r = utils.extract_feet_contacts(
                        x, [3, 4], [7, 8], velfactor=0.02
                    )
                    X.append(anim.pos[i : i + window])
                    Q.append(anim.quats[i : i + window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

            break 

    X = np.asarray(X)
    Q = np.asarray(Q)
    import pdb 
    pdb.set_trace() 
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r, seq_names

# Added by Jiaman 
def axisangle2matrots(axisangle):
    '''
    :param axisangle: N*num_joints*3
    :return: N*num_joints*9
    '''
    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size,-1,3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[1]):
            a = cv2.Rodrigues(axisangle[mIdx, jIdx:jIdx + 1, :].reshape(1, 3))[0]
            cur_axisangle.append(a)

        out_matrot.append(np.array(cur_axisangle).reshape([1,-1,9]))
    return np.vstack(out_matrot)

def quat_wxyz_to_xyzw(ori_quat):
    # ori_quat: T X 4/4 
    quat_w, quat_x, quat_y, quat_z = ori_quat[:, 0:1], ori_quat[:, 1:2], ori_quat[:, 2:3], ori_quat[:, 3:4]
    pred_quat = np.concatenate((quat_x, quat_y, quat_z, quat_w), axis=1)

    return pred_quat

def quat_xyzw_to_wxyz(ori_quat):
    # ori_quat: T X 4/4 
    quat_x, quat_y, quat_z, quat_w = ori_quat[:, 0:1], ori_quat[:, 1:2], ori_quat[:, 2:3], ori_quat[:, 3:4]
    pred_quat = np.concatenate((quat_w, quat_x, quat_y, quat_z), axis=1)

    return pred_quat

# Local positions should be the rest pose local joint offsets with the root translation. 
def get_behave_set(data_folder, actors, window=50, offset=20, train=True, selected_obj_list=None):
    npast = 1
    seq_names = []
    object_names = []
    betas_list = []
    gender_list = []
    X = []
    Q = []
    obj_q = []
    obj_x = []
    trans2joint_list = []

    parents = None 

    # Load good, bad split json of BEHAVE data
    # quality_json_path = os.path.join(data_folder.replace("_processed", ""), "behave-30fps-releasev1.json")
    quality_json_path = "/move/u/jiamanli/datasets/BEHAVE/fps30-error-frames.json"
    json_data = json.load(open(quality_json_path, 'r'))
    # good_seq_names = json_data['better_quality'] + json_data['jittering_seqs']
    good_seq_names = json_data['good_quality']

    seq_cnt = 0
    total_num_frames = 0
    for seq_name in good_seq_names:
        subject = seq_name.split("_")[1] 
        object_name = seq_name.split("_")[2]
        if selected_obj_list is None or (selected_obj_list is not None and object_name in selected_obj_list):
            if subject in actors:
                npz_path = os.path.join(data_folder, seq_name+".npz")
                if os.path.exists(npz_path):
                    seq_cnt += 1

                    npz_data = np.load(npz_path)

                    curr_betas = npz_data['betas'] # 10 
                    curr_gender = npz_data['gender']

                    root_orient = npz_data['root_orient'] # T X 3 
                    pose_body = npz_data['pose_body'] # T X 63
                    pose_hand = npz_data['pose_hand'] # T X 90 

                    person_pose = np.concatenate((root_orient, pose_body, pose_hand), axis=1) # T X 156
                    person_trans = npz_data['trans'] # T X 3 

                    timesteps = person_trans.shape[0]

                    total_num_frames += timesteps 

                    person_rot_mat = axisangle2matrots(person_pose.reshape(-1, 52, 3)) # T X 52 X 9 
                    curr_rot = R.from_matrix(person_rot_mat.reshape(-1, 3, 3)) # (T*52) X 3 X 3 

                    person_quat_xyzw = curr_rot.as_quat() # (T*52) X 4, x, y, z, w 

                    person_quat_wxyz = quat_xyzw_to_wxyz(person_quat_xyzw) # (T*52) X 4
                    person_quat_wxyz = person_quat_wxyz.reshape(-1, 52, 4) # T X 52 X 4 

                    rest_offsets = npz_data['rest_offsets'] # 52 X 3 
                    rest_offsets = torch.from_numpy(rest_offsets).float()[None] # 1 X 52 X 3
                    seq_rest_offsets = rest_offsets.repeat(timesteps, 1, 1).data.cpu().numpy() # T X 52 X 3 

                    seq_rest_offsets[:, 0, :] = person_trans # T X 52 X 3 
                    
                    obj_rot = npz_data['obj_rot'] # T X 3 
                    obj_trans = npz_data['obj_trans'] # T X 3 

                    obj_rot_mat = axisangle2matrots(obj_rot[np.newaxis]) # 1 X T X 9 
                    obj_rot_mat = obj_rot_mat.reshape(-1, 3, 3) # T X 3 X 3 
                    curr_obj_rot = R.from_matrix(obj_rot_mat)
                    obj_quat_xyzw = curr_obj_rot.as_quat() # T X 4 

                    obj_quat_wxyz = quat_xyzw_to_wxyz(obj_quat_xyzw) # T X 4 

                    parents = npz_data['parents']

                    trans2joint = npz_data['trans2joint'] # 1 X 3 

                    # Sliding windows
                    i = 0
                    while i + window < timesteps:
                        X.append(seq_rest_offsets[i:i+window])
                        Q.append(person_quat_wxyz[i:i+window])
                        obj_q.append(obj_quat_wxyz[i:i+window])
                        obj_x.append(obj_trans[i:i+window])
                        seq_names.append(seq_name)
                        object_names.append(seq_name.split("_")[2].replace(".npy", ""))
                        trans2joint_list.append(trans2joint[0])

                        betas_list.append(curr_betas)
                        gender_list.append(curr_gender)

                        i += offset

    print("Total number of sequences:{0}".format(seq_cnt))
    print("Total number of frames in 30fps:{0}".format(total_num_frames))
    print("Total durtaion of motion:{0} minutes".format(total_num_frames/30./60.))

    X = np.asarray(X) # rest pose offsets + root trajectory N X T X n_joints X 3 
    Q = np.asarray(Q) # Local quaternion N X T X n_joints X 4 
    obj_q = np.asarray(obj_q) # Object rotation quaternion N X T X 4 
    obj_x = np.asarray(obj_x) # Object translation N X T X 3 
    trans2joint_list = np.asarray(trans2joint_list) # N X 3 
    # Unify facing on last seed frame
    floor_z = True # If False, the floor is in y = xxx. 
    X, Q, obj_x, obj_q = utils.rotate_at_frame_w_obj(X, Q, obj_x, obj_q, trans2joint_list, parents, n_past=npast, floor_z=floor_z)

    # # # Set the first frame to xy=0/xz=0
    move_trans = X[:, 0:1, 0, :].copy() # N X 1 X 3 
    if floor_z:
        move_trans[:, :, 2] = 0
    else:
        move_trans[:, :, 1] = 0 
    X[:, :, 0, :] = X[:, :, 0, :] - move_trans
    obj_x = obj_x - move_trans 

    # Prepare object dict 
    obj_dict = {}
    obj_idx = 0
    for tmp_obj_name in object_names:
        if tmp_obj_name not in obj_dict:
            obj_dict[tmp_obj_name] = obj_idx 
            obj_idx +=1 

    # Assign object idx to each sequence 
    obj_idx_list = []
    for obj_name in object_names:
        obj_idx_list.append(obj_dict[obj_name])

    return X, Q, obj_x, obj_q, parents, seq_names, object_names, obj_idx_list, betas_list, gender_list

def get_train_stats(bvh_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print("Building the train set...")
    xtrain, qtrain, parents, _, _, _ = get_lafan1_set(
        bvh_folder, train_set, window=50, offset=20, train=True, stats=True
    )

    print("Computing stats...\n")
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(
        x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
        axis=(0, 2),
        keepdims=True,
    )
    x_std = np.std(
        x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
        axis=(0, 2),
        keepdims=True,
    )

    return x_mean, x_std, offsets

def get_train_stats_behave(data_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print("Building the train set...")
    xtrain, qtrain, parents, _ = get_behave_set(
        data_folder, train_set, window=50, offset=20, train=True, stats=True
    )

    print("Computing stats...\n")
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(
        x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
        axis=(0, 2),
        keepdims=True,
    )
    x_std = np.std(
        x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
        axis=(0, 2),
        keepdims=True,
    )

    return x_mean, x_std, offsets
