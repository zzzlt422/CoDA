import numpy as np
from sklearn.cluster import KMeans
import hdbscan
import itertools

def _inner_print(args,msg,log_file_path):
    if args.cluster_detial:
        if args.cluster_logger:
            with open(log_file_path, 'a') as f:
                f.write(msg + "\n")
        else:
            print(msg)


def split_hdbscan_kmeans(args, X, clusters, working_clusters, cur_mom_cluster, log_file_path, min_cluster_size, num_seed_candidates):
    ##########################################################################
    # Use HDBSCAN to split the parent cluster based on density, and use K-Means for partitioning.
    ##########################################################################
    _inner_print(args, f"Splitting mom {cur_mom_cluster['cluster_id']} Size {cur_mom_cluster['size']}", log_file_path)

    min_samples_split = args.min_samples
    mom_global_indices = np.where(cur_mom_cluster['points_mask'])[0]

    ##########################################################################
    # Stage 1: Use HDBSCAN density to find cluster centers
    ##########################################################################
    sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=args.min_samples, prediction_data=True)
    cur_mom_points = X[mom_global_indices]
    sub_labels = sub_clusterer.fit_predict(cur_mom_points)
    num_sub_clusters = len(np.unique(sub_labels)) - (1 if -1 in sub_labels else 0)

    # Current configuration cannot split the parent cluster
    if num_sub_clusters < 2:
        _inner_print(args, f"  - Insight failed: HDBSCAN found < 2 sub-clusters with minsize: {min_cluster_size}.", log_file_path)
        return

    # Parent cluster successfully split. Its features are replaced by two sub-clusters, and the parent is deleted.
    del clusters[cur_mom_cluster['cluster_id']]

    ##########################################################################
    # Stage 2: Find sub-cluster centers and form seed_pairs as initial K-Means seeds. Select the optimal pair based on K-Means inertia.
    ##########################################################################
    found_sub_clusters = []
    for i in range(num_sub_clusters):
        sub_mask_local = (sub_labels == i)
        sub_size = np.sum(sub_mask_local)
        points_in_sub_cluster = cur_mom_points[sub_mask_local]
        best_point_idx = np.argmax(sub_clusterer.probabilities_[sub_mask_local])
        center_coord = points_in_sub_cluster[best_point_idx]
        found_sub_clusters.append({'size': sub_size, 'center': center_coord})
    found_sub_clusters.sort(key=lambda c: c['size'], reverse=True)

    # Obtain candidate seed pairs
    actual_num_candidates = min(num_seed_candidates, num_sub_clusters)
    candidate_indices = range(actual_num_candidates)
    seed_pairs = list(itertools.combinations(candidate_indices, 2))

    best_kmeans_result = None
    min_inertia = float('inf')
    best_indices = None

    for idx_pair in seed_pairs:
        idx1, idx2 = idx_pair
        center1 = found_sub_clusters[idx1]['center']
        center2 = found_sub_clusters[idx2]['center']
        initial_seeds = np.array([center1, center2])

        # Partition the parent cluster using K-Means with the current seed pair.
        kmeans = KMeans(n_clusters=2, init=initial_seeds, n_init=1, random_state=0).fit(cur_mom_points)

        if kmeans.inertia_ < min_inertia:
            min_inertia = kmeans.inertia_
            best_kmeans_result = kmeans
            best_indices = idx_pair

    _inner_print(args, f"  - Insight successful: compose with {best_indices} with inertia: {min_inertia}.", log_file_path)

    final_sub_labels = best_kmeans_result.labels_
    final_centers = best_kmeans_result.cluster_centers_

    temp_labels = np.full(X.shape[0], -1)
    temp_labels[cur_mom_cluster['points_mask']] = final_sub_labels

    for i in range(2):
        new_id = args.next_new_id
        args.next_new_id += 1

        mask = (temp_labels == i)
        size = np.sum(mask)

        # This center is temporarily the geometric mean of the sub-cluster; it will be replaced by the nearest real data point in CoDA_main.py
        center = final_centers[i]

        clusters[new_id] = {
            'cluster_id': new_id,
            'center': center,
            'size': size,
            'points_mask': mask,
            'origin': 'hdbscan_kmeans_split'
        }
        _inner_print(args, f"  - Added hybrid sub-cluster {new_id} with size {size}.", log_file_path)

        # If sub-clusters are splittable, add them to working_clusters
        if size >= min_cluster_size * 2:
            working_clusters.append(clusters[new_id])



def hdbscan_post(args, M, initial_centers,cluster_labels,X, log_file_path, clusterer):

    ipc = args.IPC
    min_cluster_size = args.min_cluster_size

    args.next_new_id = M # Unique ID for adding new sub-centers

    # Outlier threshold parameter to trigger Strategy 2 or 3
    if ipc > 50:
        noise_pool_safety_factor = 1
    elif ipc > 10:
        noise_pool_safety_factor=3
    else:
        noise_pool_safety_factor=5

    num_seed_candidates = args.num_seed_candidates

    clusters = {}
    for cluster_id in range(M):
        mask = (cluster_labels == cluster_id)
        clusters[cluster_id] = {
            'cluster_id': cluster_id,
            'center': initial_centers[cluster_id],
            'size': np.sum(mask),
            'points_mask': mask,
            'origin': 'hdbscan_initial'
        }

    ##########################################################################
    # Surplus of effective centers
    # Select the representative samples corresponding to the IPC largest clusters to form the ffnal set
    ##########################################################################
    if M > ipc :
        all_clusters_list = sorted(list(clusters.values()), key=lambda c: c['size'], reverse=True)
        clusters_to_keep = all_clusters_list[:ipc]
        kept_sizes = [c['size'] for c in clusters_to_keep]
        _inner_print(args,f"M > ipc, keep {ipc} centers with each size ({sum(kept_sizes)}/{X.shape[0]}) were:\n{kept_sizes}",log_file_path)
        final_centers = {c['cluster_id']: c for c in clusters_to_keep}
        return final_centers

    ##########################################################################
    # List of dictionaries for splittable parent clusters
    # A cluster is considered effectively splittable only if its size is at least 2 * min_cluster_size
    ##########################################################################
    working_clusters = [info for info in clusters.values() if info['size'] >= min_cluster_size * 2]

    while len(clusters) < ipc and working_clusters:
        working_clusters.sort(key=lambda c: c['size'], reverse=True)
        cur_mom_cluster = working_clusters.pop(0)

        ##########################################################################
        # Strategy 1: SplitCluster。
        # Attempt to split the current parent cluster
        ##########################################################################
        split_hdbscan_kmeans(args, X, clusters, working_clusters, cur_mom_cluster, log_file_path, min_cluster_size, num_seed_candidates)

    if len(clusters) > ipc:
        _inner_print(args, f"After splitting, total clusters ({len(clusters)}) > ipc ({ipc}). Pruning...",log_file_path)
        all_clusters_list = sorted(list(clusters.values()), key=lambda c: c['size'], reverse=True)
        clusters = {c['cluster_id']: c for c in all_clusters_list[:ipc]}

    ##########################################################################
    # Activate Strategy 2 or 3 to complete the remaining tasks from Strategy 1.
    ##########################################################################
    if len(clusters) < ipc:
        num_to_create = ipc - len(clusters)
        _inner_print(args,f"After splitting, total clusters ({len(clusters)}) < ipc ({ipc}). Filling remaining {ipc-len(clusters)} centers with KMeans on noise.",log_file_path)

        # Collect all current outliers ignored by HDBSCAN
        assigned_points_mask = np.zeros(X.shape[0], dtype=bool)
        for cluster_info in clusters.values():
            assigned_points_mask |= cluster_info['points_mask']
        noise_indices = np.where(~assigned_points_mask)[0]
        pool_indices = [noise_indices]

        ##########################################################################
        # Decide whether to proceed to Strategy 2 or 3 based on the number of outliers
        ##########################################################################

        if len(noise_indices) < num_to_create * noise_pool_safety_factor:
            ##########################################################################
            # Too few outliers; enabling Strategy 3: ForcedSplit
            ##########################################################################
            _inner_print(args,f"  - Pure noise pool ({len(noise_indices)}) is small. ",log_file_path)

            # Recursively split parent clusters until len(clusters) is sufficient
            working_clusters = [info for info in clusters.values() if info['size'] >= min_cluster_size]
            while(len(clusters) < ipc):
                working_clusters.sort(key=lambda c: c['size'], reverse=True)
                temp_next_new_id = args.next_new_id
                temp_minsize = args.min_cluster_size
                cur_mom_cluster = working_clusters[0]

                while(temp_next_new_id == args.next_new_id):
                    # Equality indicates a failed split. Reduce min_cluster_size and retry.
                    temp_minsize*=0.75
                    split_hdbscan_kmeans(args, X, clusters, working_clusters, cur_mom_cluster, log_file_path, max(int(temp_minsize), 2), num_seed_candidates)
                    if temp_minsize <= 2:
                        break
                working_clusters.pop(0)

        else:
            ##########################################################################
            # Sufficient outliers; enabling Strategy 2: Clustering Outliers with one-step KMeans
            ##########################################################################
            _inner_print(args, f"  - Pure noise pool ({len(noise_indices)}) is sufficient. Using pure noise to kmeans.",log_file_path)

            final_pool_indices = np.unique(np.concatenate(pool_indices))
            final_pool_for_kmeans = X[final_pool_indices]
            kmeans_final = KMeans(n_clusters=num_to_create, random_state=0, n_init='auto').fit(final_pool_for_kmeans)
            new_centers = kmeans_final.cluster_centers_
            new_labels = kmeans_final.labels_
            for i in range(num_to_create):
                new_id = args.next_new_id
                args.next_new_id += 1
                local_mask_in_pool = (new_labels == i)
                global_indices_for_new_cluster = final_pool_indices[local_mask_in_pool]
                mask = np.zeros(X.shape[0], dtype=bool)
                mask[global_indices_for_new_cluster] = True
                size = len(global_indices_for_new_cluster)
                clusters[new_id] = {
                    'cluster_id': new_id,
                    'center': new_centers[i],
                    'size': size,
                    'points_mask': mask,
                    'origin': 'kmeans_outliers'
                }
                _inner_print(args,f"i:{i} size:{size}",log_file_path)

    assert len(clusters) == ipc, f"Logic error: len(centers) here should be same as ipc, but got len:{len(clusters)} and ipc:{ipc}"
    final_sizes = [info['size'] for info in clusters.values()]
    final_sizes.sort(reverse=True)
    sum_of_points_in_clusters = sum(final_sizes)
    _inner_print(args,f"Final clusters with each size ({sum_of_points_in_clusters}/{X.shape[0]} total points):\n{final_sizes}",log_file_path)

    return clusters





