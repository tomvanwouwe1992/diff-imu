import torch
import numpy as np
import os

def numpy2storage(labels, data, storage_file):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    f.write('name %s\n' % storage_file)
    f.write('datacolumns %d\n' % data.shape[1])
    f.write('datarows %d\n' % data.shape[0])
    f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()



file_name = "/home/tom/motion-diffusion-model/save/trying_stuff/sample"
sample = torch.load(file_name)
sample.permute(0,3,1,2)
# sample.

labels = ['time','pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_rot', 'pelvis_tilt',
          'hip_flexion_l', 'hip_flexion_r',
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
for i in range(sample.size()[0]):
    time = np.reshape(np.linspace(0,3,60),(60,1))
    sample_pelvis_rotation = 0*sample[i,0,:3,:].cpu().numpy().T # still wrong
    sample_pelvis_translation = sample[i,14,:3,:].cpu().numpy().T
    sample_pose = sample[i, 15:, 0, :].cpu().numpy().T
    data = np.concatenate((time,sample_pelvis_translation, sample_pelvis_rotation, sample_pose),1)
    numpy2storage(labels,data,os.path.join("/home/tom/motion-diffusion-model/save/trying_stuff/sample" + str(i) +  ".mot"))

print(sample)