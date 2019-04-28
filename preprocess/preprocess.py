import os
import glob
import numpy as np
import scipy.io
import struct
import array

dataset_dir = '/home/alec/Documents/Hand-Pointnet/data/cvpr15_MSRAHandGestureDB/'
save_dir = './'
subject_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
gesture_names = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']

JOINT_NUM = 21
SAMPLE_NUM = 1024
sample_num_level1 = 512
sample_num_level2 = 128
msra_valid = scipy.io.loadmat('/home/alec/Documents/Hand-Pointnet/preprocess/msra_valid.mat')['msra_valid']

for sub_idx in range(len(subject_names)):
    os.makedirs(os.path.join(save_dir, subject_names[sub_idx]), exist_ok=True)
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
        # print(save_gesture_dir)
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
            with open(os.path.join(gesture_dir, '{:06}'.format(frm_idx) + '_depth.bin'), 'rb') as f:
                data_array = np.fromfile(f, np.int32, count=6)
                img_width, img_height, bb_left, bb_top, bb_right, bb_bottom = data_array
                bb_width = bb_right - bb_left
                bb_height = bb_bottom - bb_top
                valid_pixel_num = bb_width * bb_height
                hand_depth = np.fromfile(f, np.float32).reshape(bb_width, bb_height).T
                print(hand_depth.shape)
            
            fFocal_MSRA_ = 241.42  # mm
            hand_3d = np.zeros((valid_pixel_num, 3))
            for i in range(bb_height):
                for j in range(bb_width):
                    idx = j * bb_height + i
                    hand_3d[idx, 0] = -(img_width/2 - (j+bb_left))*hand_depth[i, j] / fFocal_MSRA_
                    hand_3d[idx, 1] = (img_height/2 - (i + bb_top))*hand_depth[i, j] / fFocal_MSRA_
                    hand_3d[idx, 2] = hand_depth[i, j]
            
            valid_idx = any((hand_3d != 0))
            print(valid_idx.shape)
            raise Exception
            # hand_points = hand_3d[hand_3d != 0, :]


#         for frm_idx = 1:length(depth_files)
            
#             %% 2.2 convert depth to xyz
#             fFocal_MSRA_ = 241.42;	% mm
#             hand_3d = zeros(valid_pixel_num,3);
#             for ii=1:bb_height
#                 for jj=1:bb_width
#                     idx = (jj-1)*bb_height+ii;
#                     hand_3d(idx, 1) = -(img_width/2 - (jj+bb_left-1))*hand_depth(ii,jj)/fFocal_MSRA_;
#                     hand_3d(idx, 2) = (img_height/2 - (ii+bb_top-1))*hand_depth(ii,jj)/fFocal_MSRA_;
#                     hand_3d(idx, 3) = hand_depth(ii,jj);
#                 end
#             end

#             valid_idx = 1:valid_pixel_num;
#             valid_idx = valid_idx(hand_3d(:,1)~=0 | hand_3d(:,2)~=0 | hand_3d(:,3)~=0);
#             hand_points = hand_3d(valid_idx,:);

#             jnt_xyz = squeeze(gt_wld(frm_idx,:,:));
           
#             %% 2.3 create OBB
#             [coeff,score,latent] = princomp(hand_points);
#             if coeff(2,1)<0
#                 coeff(:,1) = -coeff(:,1);
#             end
#             if coeff(3,3)<0
#                 coeff(:,3) = -coeff(:,3);
#             end
#             coeff(:,2)=cross(coeff(:,3),coeff(:,1));

#             ptCloud = pointCloud(hand_points);

#             hand_points_rotate = hand_points*coeff;

#             %% 2.4 sampling
#             if size(hand_points,1)<SAMPLE_NUM
#                 tmp = floor(SAMPLE_NUM/size(hand_points,1));
#                 rand_ind = [];
#                 for tmp_i = 1:tmp
#                     rand_ind = [rand_ind 1:size(hand_points,1)];
#                 end
#                 rand_ind = [rand_ind randperm(size(hand_points,1), mod(SAMPLE_NUM, size(hand_points,1)))];
#             else
#                 rand_ind = randperm(size(hand_points,1),SAMPLE_NUM);
#             end
#             hand_points_sampled = hand_points(rand_ind,:);
#             hand_points_rotate_sampled = hand_points_rotate(rand_ind,:);
            
#             %% 2.5 compute surface normal
#             normal_k = 30;
#             normals = pcnormals(ptCloud, normal_k);
#             normals_sampled = normals(rand_ind,:);

#             sensorCenter = [0 0 0];
#             for k = 1 : SAMPLE_NUM
#                p1 = sensorCenter - hand_points_sampled(k,:);
#                % Flip the normal vector if it is not pointing towards the sensor.
#                angle = atan2(norm(cross(p1,normals_sampled(k,:))),p1*normals_sampled(k,:)');
#                if angle > pi/2 || angle < -pi/2
#                    normals_sampled(k,:) = -normals_sampled(k,:);
#                end
#             end
#             normals_sampled_rotate = normals_sampled*coeff;

#             %% 2.6 Normalize Point Cloud
#             x_min_max = [min(hand_points_rotate(:,1)), max(hand_points_rotate(:,1))];
#             y_min_max = [min(hand_points_rotate(:,2)), max(hand_points_rotate(:,2))];
#             z_min_max = [min(hand_points_rotate(:,3)), max(hand_points_rotate(:,3))];

#             scale = 1.2;
#             bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
#             bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
#             bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
#             max_bb3d_len = bb3d_x_len;

#             hand_points_normalized_sampled = hand_points_rotate_sampled/max_bb3d_len;
#             if size(hand_points,1)<SAMPLE_NUM
#                 offset = mean(hand_points_rotate)/max_bb3d_len;
#             else
#                 offset = mean(hand_points_normalized_sampled);
#             end
#             hand_points_normalized_sampled = hand_points_normalized_sampled - repmat(offset,SAMPLE_NUM,1);

#             %% 2.7 FPS Sampling
#             pc = [hand_points_normalized_sampled normals_sampled_rotate];
#             % 1st level
#             sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1)';
#             other_idx = setdiff(1:SAMPLE_NUM, sampled_idx_l1);
#             new_idx = [sampled_idx_l1 other_idx];
#             pc = pc(new_idx,:);
#             % 2nd level
#             sampled_idx_l2 = farthest_point_sampling_fast(pc(1:sample_num_level1,1:3), sample_num_level2)';
#             other_idx = setdiff(1:sample_num_level1, sampled_idx_l2);
#             new_idx = [sampled_idx_l2 other_idx];
#             pc(1:sample_num_level1,:) = pc(new_idx,:);
            
#             %% 2.8 ground truth
#             jnt_xyz_normalized = (jnt_xyz*coeff)/max_bb3d_len;
#             jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);

#             Point_Cloud_FPS(frm_idx,:,:) = pc;
#             Volume_rotate(frm_idx,:,:) = coeff;
#             Volume_length(frm_idx) = max_bb3d_len;
#             Volume_offset(frm_idx,:) = offset;
#             Volume_GT_XYZ(frm_idx,:,:) = jnt_xyz_normalized;
#         end
#         % 3. save files
#         save([save_gesture_dir '/Point_Cloud_FPS.mat'],'Point_Cloud_FPS');
#         save([save_gesture_dir '/Volume_rotate.mat'],'Volume_rotate');
#         save([save_gesture_dir '/Volume_length.mat'],'Volume_length');
#         save([save_gesture_dir '/Volume_offset.mat'],'Volume_offset');
#         save([save_gesture_dir '/Volume_GT_XYZ.mat'],'Volume_GT_XYZ');
#         save([save_gesture_dir '/valid.mat'],'valid');
#     end
# end