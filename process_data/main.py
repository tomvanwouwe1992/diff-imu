import numpy as np
# import opensim
import os
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import multiprocessing
import itertools
import pandas
from scipy.interpolate import interp1d
import math, pickle

def execute_body_kinematics(subject,path_motion_data):

    opensim.Logger.setLevelString('error')
    path_motion_data_subject = os.path.join(path_motion_data, subject)
    path_opensim_model = os.path.join(path_motion_data_subject, 'osim_results', 'Models',
                                      'optimized_scale_and_markers.osim')
    path_inverse_kinematics = os.path.join(path_motion_data_subject, 'osim_results', 'IK')
    inverse_kinematics_all_files = os.listdir(path_inverse_kinematics)
    motion_ext = ['.mot']
    inverse_kinematics_motion_files = [file_name for file_name in inverse_kinematics_all_files if
                                       any(sub in file_name for sub in motion_ext)]

    opensim_model = opensim.Model(path_opensim_model)

    ## body kinematics
    opensim_body_kinematics = opensim.BodyKinematics()
    opensim_model.addAnalysis(opensim_body_kinematics)
    opensim_model.initSystem()
    opensim_analyze_tool = opensim.AnalyzeTool(opensim_model)
    opensim_analyze_tool.setLoadModelAndInput(True)
    
    for trial in inverse_kinematics_motion_files:
        path_to_file = os.path.join(path_motion_data_subject, 'osim_results', 'BK',
                                    trial[:-7] + '_BodyKinematics_pos_global.sto')
        if os.path.exists(path_to_file) == False:
            path_inverse_kinematics_trial = os.path.join(path_inverse_kinematics, trial)
            table_inverse_kinematics_trial = opensim.TimeSeriesTable(path_inverse_kinematics_trial)
            time_vector_inverse_kinematics_trial_position = np.asarray(
                table_inverse_kinematics_trial.getIndependentColumn())
            path_results_folder = os.path.join(path_motion_data_subject, 'osim_results', 'BK')


            opensim_analyze_tool.setResultsDir(path_results_folder)
            opensim_analyze_tool.setName(trial[:-7])
            opensim_analyze_tool.setCoordinatesFileName(path_inverse_kinematics_trial)
            opensim_analyze_tool.setStartTime(time_vector_inverse_kinematics_trial_position[0])
            opensim_analyze_tool.setFinalTime(time_vector_inverse_kinematics_trial_position[-1])
            opensim_analyze_tool.run()

            print('Subject ' + subject + ' trial ' + trial + ' processed.')


def transform_kinematics_output_to_training_data_representation(subject, path_motion_data):
    bodies = ['pelvis',
              'femur_r', 'tibia_r', 'talus_r',
              'femur_l', 'tibia_l', 'talus_l',
              'torso',
              'humerus_r', 'radius_r', 'hand_r',
              'humerus_l', 'radius_l', 'hand_l']
    joints = ['hip_flexion_l', 'hip_flexion_r',
              'hip_adduction_l', 'hip_adduction_r',
              'hip_rotation_l', 'hip_rotation_r',
              'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r',
              'subtalar_angle_l', 'subtalar_angle_r',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
              'arm_flex_l', 'arm_flex_r',
              'arm_add_l', 'arm_add_r',
              'arm_rot_l', 'arm_rot_r',
              'elbow_flex_l', 'elbow_flex_r',
              'pro_sup_l', 'pro_sup_r']

    opensim.Logger.setLevelString('error')
    path_motion_data_subject = os.path.join(path_motion_data, subject)
    path_motion_data_training_representation_subject = os.path.join(path_motion_data, subject, 'diffusion_data')

    if os.path.isdir(os.path.join(path_motion_data, subject, 'diffusion_data')) == False:
        os.makedirs(os.path.join(path_motion_data, subject, 'diffusion_data'))
    path_inverse_kinematics = os.path.join(path_motion_data_subject, 'osim_results', 'IK')
    inverse_kinematics_all_files = os.listdir(path_inverse_kinematics)
    inverse_kinematics_files = [file_name for file_name in inverse_kinematics_all_files if
                             any(sub in file_name for sub in '.mot')]
    path_body_kinematics = os.path.join(path_motion_data_subject, 'osim_results', 'BK')
    body_kinematics_all_files = os.listdir(path_body_kinematics)
    motion_ext = ['.sto']
    body_kinematics_files = [file_name for file_name in body_kinematics_all_files if
                                       any(sub in file_name for sub in motion_ext)]


    for trial in body_kinematics_files:
        labels = ['time']
        if 'BodyKinematics_pos' in trial and 'larger' not in trial:
            path_body_kinematics_trial = os.path.join(path_body_kinematics, trial)

            # if os.path.exists(os.path.join(path_motion_data_training_representation_subject, trial[:-15] + '.pkl')) == False:

            table_body_kinematics_trial = opensim.TimeSeriesTable(path_body_kinematics_trial)

            matrix_body_kinematics_position = table_body_kinematics_trial.getMatrix().to_numpy()
            time_vector_body_kinematics_position = np.asarray(table_body_kinematics_trial.getIndependentColumn())
            column_labels_body_kinematics_position = table_body_kinematics_trial.getColumnLabels()

            column_labels_body_kinematics_position = table_body_kinematics_trial.getColumnLabels()
            if table_body_kinematics_trial.getTableMetaDataAsString('inDegrees') == 'yes':
                angle_multiplier = np.pi / 180
            else:
                angle_multiplier = 1

            euler_angles = np.zeros((np.shape(time_vector_body_kinematics_position)[0], 3))

            data_matrix = np.zeros((np.shape(time_vector_body_kinematics_position)[0], len(bodies) * (9) + len(joints) + 1 + 3))
            data_matrix[:, 0] = time_vector_body_kinematics_position

            for i, body in enumerate(bodies):
                index_Ox = column_labels_body_kinematics_position.index(body + '_Ox')
                index_Oy = column_labels_body_kinematics_position.index(body + '_Oy')
                index_Oz = column_labels_body_kinematics_position.index(body + '_Oz')
                euler_angles[:, 0] = matrix_body_kinematics_position[:, index_Ox]
                euler_angles[:, 1] = matrix_body_kinematics_position[:, index_Oy]
                euler_angles[:, 2] = matrix_body_kinematics_position[:, index_Oz]
                euler_angles = angle_multiplier * euler_angles
                rotation_body = Rotation.from_euler('xyz', euler_angles)
                rotation_body_matrix = rotation_body.as_matrix()
                rotation_body_matrix_flattened = np.reshape(rotation_body_matrix,(np.shape(time_vector_body_kinematics_position)[0],9))
                rotation_body_quaternion = rotation_body.as_quat()
                # labels.append(body + '_euler_X')
                # labels.append(body + '_euler_Y')
                # labels.append(body + '_euler_Z')
                labels.append(body + '_rotation_matrix_1')
                labels.append(body + '_rotation_matrix_2')
                labels.append(body + '_rotation_matrix_3')
                labels.append(body + '_rotation_matrix_4')
                labels.append(body + '_rotation_matrix_5')
                labels.append(body + '_rotation_matrix_6')
                labels.append(body + '_rotation_matrix_7')
                labels.append(body + '_rotation_matrix_8')
                labels.append(body + '_rotation_matrix_9')

                # labels.append(body + '_quat_1')
                # labels.append(body + '_quat_2')
                # labels.append(body + '_quat_3')
                # labels.append(body + '_quat_4')

                data_matrix[:, i*(9) + 1:(i + 1)*(9) + 1] = rotation_body_matrix_flattened

            index_ik = inverse_kinematics_files.index(trial[:-30] + '_ik.mot')

            path_inverse_kinematics_trial = os.path.join(path_inverse_kinematics, trial[:-30] + '_ik.mot')
            table_inverse_kinematics_trial = opensim.TimeSeriesTable(path_inverse_kinematics_trial)

            matrix_inverse_kinematics_position = table_inverse_kinematics_trial.getMatrix().to_numpy()
            column_labels_inverse_kinematics_position = table_inverse_kinematics_trial.getColumnLabels()

            for i, joint in enumerate(joints):
                index = column_labels_inverse_kinematics_position.index(joint)
                labels.append(joint)
                joint_position = matrix_inverse_kinematics_position[:, index]
                data_matrix[:, (len(bodies))*(9) + 1 + i] = joint_position


            pelvis_position = np.zeros((np.shape(time_vector_body_kinematics_position)[0], 3))
            index_Ox = column_labels_body_kinematics_position.index('pelvis_X')
            index_Oy = column_labels_body_kinematics_position.index('pelvis_Y')
            index_Oz = column_labels_body_kinematics_position.index('pelvis_Z')
            labels.append('pelvis_X')
            labels.append('pelvis_Y')
            labels.append('pelvis_Z')
            pelvis_position[:, 0] = matrix_body_kinematics_position[:, index_Ox]
            pelvis_position[:, 1] = matrix_body_kinematics_position[:, index_Oy]
            pelvis_position[:, 2] = matrix_body_kinematics_position[:, index_Oz]
            data_matrix[:,-3:] = pelvis_position

            data_frame = pandas.DataFrame(data_matrix,index = None, columns = labels)


            data_frame.to_pickle(os.path.join(path_motion_data_training_representation_subject, trial[:-15] + '.pkl'))
            print('Subject ' + subject + ' trial ' + trial + ' processed.')

def generate_training_datastructure_from_trials(subject, path_motion_data):
    path_motion_data_subject = os.path.join(path_motion_data, subject)
    path_motion_data_training_representation_subject = os.path.join(path_motion_data, subject, 'diffusion_data')
    motion_data_training_representation_all_files = os.listdir(path_motion_data_training_representation_subject)
    motion_data_training_representation_files = [file_name for file_name in motion_data_training_representation_all_files if
                                any(sub in file_name for sub in '.pkl')]
    clip_lengths = []
    resampling_freq = 20
    resampling_period = 1/20
    clip_length = 3
    number_of_frames_per_clip = clip_length*resampling_freq
    clips = []
    for trial in motion_data_training_representation_files:
        data_frame = pandas.read_pickle(os.path.join(path_motion_data_training_representation_subject,trial))
        time_vector = data_frame['time'].to_numpy()
        data_matrix = data_frame.to_numpy()

        data_matrix_interp = interp1d(time_vector,data_matrix.T,kind='cubic')
        time_vector_downsampled_last = math.floor(time_vector[-1] / resampling_period) * resampling_period

        time_vector_downsampled = np.around(np.linspace(0,time_vector_downsampled_last,int(time_vector_downsampled_last*20) + 1),2)
        data_matrix_downsampled = data_matrix_interp(time_vector_downsampled).T

        # Split in 3s clips

        number_of_clips = math.floor(time_vector_downsampled[-1] / clip_length)
        for i in range(number_of_clips):
            clip_start_index = i * (number_of_frames_per_clip)
            clip_end_index = (i + 1) * (number_of_frames_per_clip) + 1
            clip_matrix = data_matrix_downsampled[clip_start_index:clip_end_index,:]
            clip_dataframe = pandas.DataFrame(clip_matrix, index=None, columns=data_frame.columns)
            clips.append(clip_dataframe)


    return clips





if __name__=="__main__":

    run_body_kinematics = False
    body_kinematics_output_to_training_data_representation = False
    generate_training_datastructure = True
    zero_offset_pelvisX_pelvisZ = True

    bodies = ['pelvis',
              'femur_r', 'tibia_r', 'talus_r',
              'femur_l', 'tibia_l', 'talus_l',
              'torso',
              'humerus_r', 'radius_r', 'hand_r',
              'humerus_l', 'radius_l', 'hand_l']
    joints = ['hip_flexion_l', 'hip_flexion_r',
              'hip_adduction_l', 'hip_adduction_r',
              'hip_rotation_l', 'hip_rotation_r',
              'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r',
              'subtalar_angle_l', 'subtalar_angle_r',
              'lumbar_extension', 'lumbar_bending','lumbar_rotation',
              'arm_flex_l', 'arm_flex_r',
              'arm_add_l', 'arm_add_r',
              'arm_rot_l', 'arm_rot_r',
              'elbow_flex_l', 'elbow_flex_r',
              'pro_sup_l', 'pro_sup_r']

    path_main = os.path.dirname(os.path.dirname(os.getcwd()))
    path_motion_data = os.path.join(path_main,'diff-imu-input', 'processMotionData','motionData')

    if run_body_kinematics == True:
        opensim.Logger.setLevelString('error')
        subjects = os.listdir(path_motion_data)
        iterable = []
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(execute_body_kinematics, zip(subjects, itertools.repeat(path_motion_data)))
        pool.close()
        pool.join()
        print('done')

    if body_kinematics_output_to_training_data_representation == True:
        subjects = os.listdir(path_motion_data)
        iterable = []
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(transform_kinematics_output_to_training_data_representation, zip(subjects, itertools.repeat(path_motion_data)))
        pool.close()
        pool.join()

    if generate_training_datastructure == True:
        subjects = os.listdir(path_motion_data)
        all_clip_lengths = []
        for subject in subjects:
            clip_lengths = generate_training_datastructure_from_trials(subject,path_motion_data)
            all_clip_lengths.append(clip_lengths)
        clips = [item for sublist in all_clip_lengths for item in sublist]

        if zero_offset_pelvisX_pelvisZ == True:
            for clip in clips:
                clip.pelvis_X -= clip.pelvis_X.to_numpy()[0]
                clip.pelvis_Z -= clip.pelvis_Z.to_numpy()[0]
                clip.time -= clip.time.to_numpy()[0]

        clips_numpy = []

        for clip in clips:
            clips_numpy.append(clip.to_numpy())

        with open('CMU_motion_diffusion_dataframe.pkl', 'wb') as f:
            pickle.dump(clips, f)

        with open('CMU_motion_diffusion_numpy.pkl', 'wb') as f:
            pickle.dump(clips_numpy, f)

    print('CMU')









# for subject in subjects:
#     path_motion_data_subject =  os.path.join(path_motion_data, subject)
#     path_opensim_model = os.path.join(path_motion_data_subject, 'osim_results', 'Models', 'optimized_scale_and_markers.osim')
#     path_inverse_kinematics = os.path.join(path_motion_data_subject, 'osim_results','IK')
#     inverse_kinematics_all_files = os.listdir(path_inverse_kinematics)
#     motion_ext = ['.mot']
#     inverse_kinematics_motion_files = [file_name for file_name in inverse_kinematics_all_files if any(sub in file_name for sub in motion_ext)]
#     for trial in inverse_kinematics_motion_files:
#         path_to_file =  os.path.join(path_motion_data_subject, 'osim_results', 'BK', trial[:-7] + '_BodyKinematics_pos_global.sto')
#         if os.path.exists(path_to_file) == False:
#             path_inverse_kinematics_trial = os.path.join(path_inverse_kinematics, trial)
#             table_inverse_kinematics_trial = opensim.TimeSeriesTable(path_inverse_kinematics_trial)
#             time_vector_inverse_kinematics_trial_position = np.asarray(table_inverse_kinematics_trial.getIndependentColumn())
#             path_results_folder = os.path.join(path_motion_data_subject, 'osim_results', 'BK')
#
#             opensim_model = opensim.Model(path_opensim_model)
#
#             ## body kinematics
#             opensim_body_kinematics = opensim.BodyKinematics()
#             opensim_model.addAnalysis(opensim_body_kinematics)
#             opensim_model.initSystem()
#             opensim_analyze_tool = opensim.AnalyzeTool(opensim_model)
#             opensim_analyze_tool.setLoadModelAndInput(True)
#             opensim_analyze_tool.setResultsDir(path_results_folder)
#             opensim_analyze_tool.setName(trial[:-7])
#             opensim_analyze_tool.setCoordinatesFileName(path_inverse_kinematics_trial)
#             opensim_analyze_tool.setStartTime(time_vector_inverse_kinematics_trial_position[0])
#             opensim_analyze_tool.setFinalTime(time_vector_inverse_kinematics_trial_position[-1])
#             opensim_analyze_tool.run()
#
#             print('Subject ' + subject + ' trial ' + trial + ' processed.')

#
#
# path_body_kinematics_position = os.path.join(path_results_folder,trial + '_BodyKinematics_pos_global.sto')
#
# # Create time-series table with coordinate values.
# table_body_kinematics_position = opensim.TimeSeriesTable(path_body_kinematics_position)
# matrix_body_kinematics_position = table_body_kinematics_position.getMatrix().to_numpy()
# time_vector_body_kinematics_position = np.asarray(table_body_kinematics_position.getIndependentColumn())
# column_labels_body_kinematics_position = table_body_kinematics_position.getColumnLabels()
# index = table_body_kinematics_position.getColumnIndex(column_labels_body_kinematics_position[3])
#
# if  table_body_kinematics_position.getTableMetaDataAsString('inDegrees') == 'yes':
#     angle_multiplier = np.pi/180
# else:
#     angle_multiplier = 1
#
#
# euler_angles = np.zeros((np.shape(time_vector_body_kinematics_position)[0], 3))
# for body in bodies:
#     fig, axs = plt.subplots(1, 3, sharex=True)
#     fig.suptitle(body)
#
#
#     index_Ox = column_labels_body_kinematics_position.index(body + '_Ox')
#     index_Oy = column_labels_body_kinematics_position.index(body + '_Oy')
#     index_Oz = column_labels_body_kinematics_position.index(body + '_Oz')
#     euler_angles[:, 0] = matrix_body_kinematics_position[:, index_Ox]
#     euler_angles[:, 1] = matrix_body_kinematics_position[:, index_Oy]
#     euler_angles[:, 2] = matrix_body_kinematics_position[:, index_Oz]
#     euler_angles = angle_multiplier * euler_angles
#     rotation_body = Rotation.from_euler('xyz',euler_angles)
#     rotation_body_matrix = rotation_body.as_matrix()
#     rotation_body_quaternion = rotation_body.as_quat()
#
#     titles_dim = ['euler', 'rotation matrix', 'quaternion']
#     for i, ax in enumerate(axs.flat):
#         if i == 0:
#             ax.plot(time_vector_body_kinematics_position, euler_angles)
#         elif i == 1:
#             rotation_body_matrix_reshaped = np.reshape(rotation_body_matrix, (np.shape(time_vector_body_kinematics_position)[0],9))
#             ax.plot(time_vector_body_kinematics_position, rotation_body_matrix_reshaped)
#         elif i == 2:
#             ax.plot(time_vector_body_kinematics_position, rotation_body_quaternion)
#
#         ax.set_title(titles_dim[i])
#
# plt.show()
#



