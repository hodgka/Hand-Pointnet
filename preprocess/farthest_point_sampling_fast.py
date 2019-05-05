import numpy as np

def farthest_point_sampling_fast(point_cloud, sample_num):
    pc_num = point_cloud.shape[0]
    if pc_num <= sample_num:
        sampled_idx = np.arange(pc_num)
        sampled_idx = np.append(sampled_idx, np.random.choice(sampled_idx, sample_num-pc_num, replace=True))
    else:
        sampled_idx = np.zeros((sample_num), dtype=np.int32)
        sampled_idx[0] = np.random.choice(np.arange(pc_num))
        # print(point_cloud)
        cur_sample = point_cloud[sampled_idx[0]]
        min_dist = np.sum((point_cloud - cur_sample)**2, axis=1)

        # print("MIN DIST", min_dist)
        for cur_sample_idx in range(1, sample_num):
            sampled_idx[cur_sample_idx] = np.argmax(min_dist)
            # print("SAMPLING:", sampled_idx)
            if cur_sample_idx < sample_num:
                valid_idx = min_dist > 1e-8
                diff = point_cloud[min_dist>1e-8] - np.tile(point_cloud[sampled_idx[cur_sample_idx], :], np.sum(valid_idx)).reshape(-1, 3)
                min_dist[valid_idx] = np.minimum(min_dist[valid_idx], np.sum(diff*diff, axis=1))
    sampled_idx = np.unique(sampled_idx).reshape(-1, 1)
    return sampled_idx


if __name__ == '__main__':
    # data = np.random.randint(-10, 10, size=(10, 3))
    data = np.array([
        [0, 0, 0], # 0
        [1, 0, 0], # 1
        [2, 0, 0], # 2
        [3, 0, 0], # 3
        [-3, 0, 0], # 4
        [-2, 0, 0], # 5
        [-1, 0, 0], # 6
    ])
    sample_num=6

    sampled = farthest_point_sampling_fast(data, sample_num)
    print("FINAL SAMPLING: ", sampled)