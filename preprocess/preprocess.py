import os
import glob
import numpy as np
import scipy.io
import struct
import array
from sklearn.decomposition import PCA
from open3d import PointCloud, Vector3dVector, orient_normals_towards_camera_location

dataset_dir = '/home/alec/Documents/3d_hand_pose/data/cvpr15_MSRAHandGestureDB/'
save_dir = '/home/alec/Documents/3d_hand_pose/data/preprocessed/'
subject_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
gesture_names = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']

JOINT_NUM = 21
SAMPLE_NUM = 1024
sample_num_level1 = 512
sample_num_level2 = 128
msra_valid = scipy.io.loadmat('/home/alec/Documents/3d_hand_pose/preprocess/msra_valid.mat')['msra_valid']

def farthest_point_sampling_fast(point_cloud, sample_num):
    pc_num = point_cloud.shape[0]

    if pc_num <= sample_num:
        sampled_idx = np.arange(pc_num)
        sampled_idx = np.append(sampled_idx, np.random.randint(0, pc_num, sample_num - pc_num))
        # sampled_idx = [sampled_idx; randi([1,pc_num],sample_num-pc_num,1)];
    else:
        sampled_idx = np.zeros((sample_num,1))
        sampled_idx[0] = np.random.randint(0, pc_num)
        cur_sample = np.tile(point_cloud[sampled_idx[0], :], (pc_num, 1))
        # sampled_idx(1) = randi([1,pc_num]);
        
        cur_sample = repmat(point_cloud(sampled_idx(1),:),pc_num,1);
        diff = point_cloud - cur_sample;
        min_dist = sum(diff.*diff, 2);

        for cur_sample_idx = 2:sample_num
            %% find the farthest point
            [~, sampled_idx(cur_sample_idx)] = max(min_dist);
            
            if cur_sample_idx < sample_num
                % update min_dist
                valid_idx = (min_dist>1e-8);
                diff = point_cloud(valid_idx,:) - repmat(point_cloud(sampled_idx(cur_sample_idx),:),sum(valid_idx),1);
                min_dist(valid_idx,:) = min(min_dist(valid_idx,:), sum(diff.*diff, 2));
            end
        end
    end
    sampled_idx = unique(sampled_idx);


for sub_idx in range(len(subject_names)):
    os.makedirs(os.path.join(save_dir, subject_names[sub_idx]), exist_ok=True)
    print(os.path.join(save_dir, subject_names[sub_idx]))
    for ges_idx in range(len(gesture_names)):
        gesture_dir = os.path.join(dataset_dir, subject_names[sub_idx], gesture_names[ges_idx])
        depth_files = glob.glob(os.path.join(gesture_dir, '*.bin'))
        with open(os.path.join(gesture_dir, 'joint.txt'), 'r') as f:
            frame_num = int(f.readline().strip())
            coords = []
            for line in f:
                coords.append(line.strip().split())
            A = np.array(coords, dtype=np.float32)
            # A = np.array(f.read().strip().split(), dtype=np.float32)
            gt_world = A.reshape(3, 21, frame_num)
            gt_world[2, :, :] *= -1
            gt_world = np.transpose(gt_world, axes=[2, 1, 0])
        
        save_gesture_dir = os.path.join(save_dir, subject_names[sub_idx], gesture_names[ges_idx])
        print(save_gesture_dir)
        os.makedirs(save_gesture_dir, exist_ok=True)
        
        Point_Cloud_FPS = np.zeros((frame_num, SAMPLE_NUM, 6))
        Volume_rotate = np.zeros((frame_num, 3, 3))
        Volume_length = np.zeros((frame_num, 1))
        Volume_offset = np.zeros((frame_num, 3))
        Volume_GT_XYZ = np.zeros((frame_num, JOINT_NUM, 3))
        valid = msra_valid[sub_idx, ges_idx]
        for frm_idx in range(len(depth_files)):
            if not valid[frm_idx]:
                continue
            
            # Read data from binary file
            with open(os.path.join(gesture_dir, '{:06}'.format(frm_idx) + '_depth.bin'), 'rb') as f:
                data_array = np.fromfile(f, np.int32, count=6)
                img_width, img_height, bb_left, bb_top, bb_right, bb_bottom = data_array
                bb_width = bb_right - bb_left
                bb_height = bb_bottom - bb_top
                valid_pixel_num = bb_width * bb_height
                hand_depth = np.fromfile(f, np.float32).reshape(bb_width, bb_height).T
            
            # convert depth to xyz
            fFocal_MSRA_ = 241.42  # mm
            hand_3d = np.zeros((valid_pixel_num, 3))
            for i in range(bb_height):
                for j in range(bb_width):
                    idx = j * bb_height + i
                    hand_3d[idx, 0] = -(img_width/2 - (j+bb_left))*hand_depth[i, j] / fFocal_MSRA_
                    hand_3d[idx, 1] = (img_height/2 - (i + bb_top))*hand_depth[i, j] / fFocal_MSRA_
                    hand_3d[idx, 2] = hand_depth[i, j]
            
            valid_idx_mask = (hand_3d[:, 0] != 0) | (hand_3d[:, 1] != 0) | (hand_3d[:, 2] != 0)

            valid_idx = np.arange(valid_pixel_num)[valid_idx_mask]
            hand_points = hand_3d[valid_idx, :]
            joint_xyz = np.squeeze(gt_world[frm_idx, :, :])
            print(hand_points.shape)


            # create OBB
            pca = PCA()
            pca.fit(hand_points)
            coeff = pca.components_
            if coeff[1, 0] < 0:
                coeff[:, 0] = -coeff[:, 0]
            if coeff[2, 2] < 0
            coeff[:, 2] = -coeff[:, 2]
            coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])

            pt_cloud = PointCloud()
            pt_cloud.points = Vector3dVector(hand_points)
            # pt_cloud = PointCloud(hand_points)
            hand_points_rotate = np.matmul(hand_points, coeff)

            vals = np.arange(hand_points.shape[0])
            if hand_points.shape[0] < SAMPLE_NUM:
                tmp = SAMPLE_NUM // hand_points.shape[0]
                rand_ind = np.tile(vals, tmp)
                remaining = SAMPLE_NUM % hand_points.shape[0]
                padding = np.random.choice(vals, remaining, False)
                rand_ind = np.append(rand_ind, padding)
            else:
                rand_ind = np.random.choice(vals, SAMPLE_NUM)

            hand_points_sampled = hand_points[rand_ind, :]
            hand_points_rotate_sampled = hand_points_rotate[rand_ind, :]


            # compute surface normals
            normal_k = 30
            estimate_normals(p, search_param = KDTreeSearchParamHybrid(radius=1, max_nn=30))
            normals = np.asarray(pt_cloud.normals)
            normals_sampled = normals[rand_ind, :]
            sensor_center = np.array([0, 0, 0])
            for k in range(SAMPLE_NUM):
                p1 = sensor_center - hand_points_sampled[k, :]
                # flip the normal vector if it is not pointing towards the sensor
                product = np.cross(p1, normals_sampled[k, :])
                angle = np.arctan2(np.norm(product), np.matmul(p1, normals[k, :]).T)
                if angle > np.pi / 2 or angle < -np.pi / 2:
                    normals_sampled[k, :] = - normals_sampled[k, :]
            
            normals_sampled_rotate = np.matmul(normals_sampled, coeff)

            # normalize point_cloud
            x_diff = max(hand_points_rotate[:, 0]) - min(hand_points_rotate[:, 0])
            y_diff = max(hand_points_rotate[:, 1]) - min(hand_points_rotate[:, 1])
            z_diff = max(hand_points_rotate[:, 2]) - min(hand_points_rotate[:, 2])
            scale = 1.2
            bb3d_x_len = scale*x_diff
            bb3d_y_len = scale*y_diff
            bb3d_z_len = scale*z_diff
            max_bb3d_len = bb3d_x_len
            hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
            if hand_points.shape[0] < SAMPLE_NUM:
                offset = np.mean(hand_points_rotate) / max_bb3d_len
            else:
                offset = np.mean(hand_points_normalized_sampled)
            
            hand_points_normalized_sampled = hand_points_normalized_sampled - np.tile(offset, (SAMPLE_NUM, 1))

            # FPS Sampling
            pc = np.concatenate((hand_points_normalized_sampled, normals_sampled_rotate))
            # 1st level
            sampled_idx_l1 = fath


            # raise Exception
            # hand_points = hand_3d[hand_3d != 0, :]


            
            hand_points_normalized_sampled = hand_points_normalized_sampled - repmat(offset,SAMPLE_NUM,1);

            %% 2.7 FPS Sampling
            pc = [hand_points_normalized_sampled normals_sampled_rotate];
            % 1st level
            sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1)';
            other_idx = setdiff(1:SAMPLE_NUM, sampled_idx_l1);
            new_idx = [sampled_idx_l1 other_idx];
            pc = pc(new_idx,:);
            % 2nd level
            sampled_idx_l2 = farthest_point_sampling_fast(pc(1:sample_num_level1,1:3), sample_num_level2)';
            other_idx = setdiff(1:sample_num_level1, sampled_idx_l2);
            new_idx = [sampled_idx_l2 other_idx];
            pc(1:sample_num_level1,:) = pc(new_idx,:);
            
            %% 2.8 ground truth
            jnt_xyz_normalized = (jnt_xyz*coeff)/max_bb3d_len;
            jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);

            Point_Cloud_FPS(frm_idx,:,:) = pc;
            Volume_rotate(frm_idx,:,:) = coeff;
            Volume_length(frm_idx) = max_bb3d_len;
            Volume_offset(frm_idx,:) = offset;
            Volume_GT_XYZ(frm_idx,:,:) = jnt_xyz_normalized;
        end
        % 3. save files
        save([save_gesture_dir '/Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
        save([save_gesture_dir '/Volume_rotate.mat'],'Volume_rotate');
        save([save_gesture_dir '/Volume_length.mat'],'Volume_length');
        save([save_gesture_dir '/Volume_offset.mat'],'Volume_offset');
        save([save_gesture_dir '/Volume_GT_XYZ.mat'],'Volume_GT_XYZ');
        save([save_gesture_dir '/valid.mat'],'valid');
    end
end