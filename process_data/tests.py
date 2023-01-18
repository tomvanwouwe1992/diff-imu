from scipy.spatial.transform import Rotation
import utils.rotation_conversions as geometry
import numpy as np
import torch


def test_diff_to_motion_file(diff_output,matrix_inverse_kinematics_position, column_labels_inverse_kinematics_position):

    time_dimension = diff_output.shape[0]

    body_orientations_as_flattened_rotation_matrix = diff_output[:, 1:1+14*9]
    joint_coordinates = diff_output[:, 1+14*9:1+14*9+25]
    pelvis_translation = diff_output[:, -3:]

    body_orientations_as_rotation_matrix = torch.from_numpy(np.reshape(body_orientations_as_flattened_rotation_matrix, (time_dimension, 14, 3, 3)))
    body_orientations_as_rotation_6d = geometry.matrix_to_rotation_6d(body_orientations_as_rotation_matrix)

    output_body_orientations_as_rotation_matrix = geometry.rotation_6d_to_matrix(body_orientations_as_rotation_6d)

    # Check whether back and forth reconstruction through 6d representation is one-to-one
    assert (np.abs((output_body_orientations_as_rotation_matrix.numpy() - body_orientations_as_rotation_matrix.numpy())).flatten() < 1e-6).all(), 'something wrong with going back and forth matrix and 6d representation of rotation'

    pelvis_rotation_as_rotation_matrix = output_body_orientations_as_rotation_matrix[:,0,:,:]

    pelvis_rotation_as_pelvis_euler = Rotation.from_matrix(pelvis_rotation_as_rotation_matrix).as_euler('ZXY')
    joint_coordinates = joint_coordinates
    data_matrix_for_motion_file = np.concatenate((pelvis_rotation_as_pelvis_euler, pelvis_translation, joint_coordinates),1)

    matrix_inverse_kinematics_position_without_mtp = np.delete(matrix_inverse_kinematics_position,[12,18],1)

    assert ((np.abs(data_matrix_for_motion_file - matrix_inverse_kinematics_position_without_mtp)).flatten() < 1e-6).all(), 'mapping to motion file is incorrect'

    return