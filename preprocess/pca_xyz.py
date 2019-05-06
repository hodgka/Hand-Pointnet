import numpy as np
import os
from sklearn.decomposition import PCA

dataset_dir = '/home/alec/Documents/3d_hand_pose/data/preprocessed/'
msra_mat_dir = '/home/alec/Documents/3d_hand_pose/preprocess/msra_valid.mat'

subject_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
gesture_names = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']

JOINT_NUM = 21
pca = PCA()
for i in range(9):

    joint_xyz = np.empty((1, 63))
    for sub_idx in range(len(subject_names)):
        for ges_idx in range(len(gesture_names)):
            gesture_dir = os.path.join(dataset_dir, subject_names[sub_idx], gesture_names[ges_idx])
            # volume_gt = np.load(os.path.join(gesture_dir, ))
            # print(gesture_dir, os.listdir(gesture_dir))
            volume_gt = np.load(os.path.join(gesture_dir, 'Volume_GT_XYZ.npy'))
            valid = np.load(os.path.join(gesture_dir, 'valid.npy'))
            tmp1 = np.transpose(volume_gt, axes=[0, 2, 1])
            tmp2 = np.reshape(tmp1, (volume_gt.shape[0], JOINT_NUM*3))
            # print(joint_xyz, tmp2.shape)
            if sub_idx != i:
                joint_xyz = np.append(joint_xyz, tmp2, axis=0)

    # joint_xyz.reshape(-1, 1)
    print(joint_xyz.shape)
    print(np.max(joint_xyz), np.min(joint_xyz))
    pca.fit(joint_xyz)
    PCA_mean_xyz = np.mean(joint_xyz, axis=0)
    pca_coeff = pca.components_
    latent = pca.explained_variance_
    save_dir = os.path.join(dataset_dir, subject_names[i])
    np.save(os.path.join(save_dir, 'PCA_mean_xyz.npy'), PCA_mean_xyz)
    np.save(os.path.join(save_dir, 'PCA_coeff.npy'), pca_coeff)
    np.save(os.path.join(save_dir, 'PCA_latent_weight.npy'), latent)

