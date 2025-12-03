
import os
import argparse
import copy
import pickle

import torch
import numpy as np
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan

from get_features import calculate_features_multiprocess
from postprocess import _inner_print, hdbscan_post
from generated import  generate_images_multi_gpu

import warnings
warnings.filterwarnings("ignore", module='sklearn')
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.")

def save_clusters(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Clusters centers saved to: {file_path}")

def get_class_info(args):
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet100':
        file_list = './misc/class100.txt'
    elif args.spec == 'imagenet1k':
        file_list = './misc/class_indices.txt'
    elif args.spec == 'IDC':
        file_list = './misc/class_IDC.txt'
    elif args.spec == 'imageA':
        file_list = './misc/imagenet-a.txt'
    elif args.spec == 'imageB':
        file_list = './misc/imagenet-b.txt'
    elif args.spec == 'imageC':
        file_list = './misc/imagenet-c.txt' 
    elif args.spec == 'imageD':
        file_list = './misc/imagenet-d.txt'
    elif args.spec == 'imageE':
        file_list = './misc/imagenet-e.txt'
    else:
        raise ValueError("Invalid spec")
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []

    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    class_id_to_name = {}
    class_names_for_saving = []
    with open('./misc/class_names.txt', 'r') as fp:
        for i, line in enumerate(fp):
            class_name = line.strip()
            if i < len(all_classes):
                class_id_to_name[all_classes[i]] = class_name
    # get the text prompts
    text_prompts = []
    for class_id in sel_classes:
        text_prompts.append(class_id_to_name[class_id])
        # only use the first part of the class name as the save name
        class_names_for_saving.append(class_id_to_name[class_id].split(',')[0].strip())

    print(f"Dataset: {args.spec}: {class_names_for_saving}")
    return sel_classes, class_labels, class_id_to_name, text_prompts, class_names_for_saving

def main(args):

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    sel_classes, class_labels, class_id_to_name, text_prompts, class_names_for_saving = get_class_info(args)

    # region Encode the original dataset using VAE.
    if args.calcu_features:
        print(f"Getting features from scratch!")
        calculate_features_multiprocess(args)
    # endregion

    # region Perform clustering in the VAE latent space to identify IPC representative samples, forming set R.
    if args.calcu_cluster:
        log_file_path = args.log_file_path
        if args.cluster_detial and args.cluster_logger:
            with open(log_file_path, 'w') as f:
                f.write("Cluster Details Log\n")
                f.write("=" * 40 + "\n")

        num_chunks = args.nclass // 10
        for chunk_id in range(args.begin_chunk_id, num_chunks):
            clusters_centers = dict()

            chunk_file_path = f"{args.features_cache_path}_{chunk_id}"
            with open(chunk_file_path, "rb") as f:
                cache_data = pickle.load(f)
            original_features_per_class = copy.deepcopy(cache_data["features"])
            original_paths = copy.deepcopy(cache_data["paths"])
            del cache_data

            for c in tqdm(original_features_per_class.keys(), desc=f"Clustering Chunk {chunk_id}"):

                final_resized_real_image_dir = os.path.join(args.save_dir, 'real_images', sel_classes[c])
                os.makedirs(final_resized_real_image_dir, exist_ok=True)

                # preprocess with StandardScaler
                scaler = StandardScaler()
                X_original_unscaled = np.stack(original_features_per_class[c])
                X = scaler.fit_transform(X_original_unscaled)

                ##########################################################################
                # preprocess with UMAP
                ##########################################################################
                umap_reducer = UMAP(n_components=50, n_neighbors=args.n_neighbors, min_dist=0.0,random_state=42)
                X_processed = umap_reducer.fit_transform(X)
                _inner_print(args,f"[Class {c}] Original feature dimension: {X.shape[1]} After UMAP: {X_processed.shape[1]}", log_file_path)

                ##########################################################################
                # Clustering with HDBSCAN to get the initial centers
                ##########################################################################
                clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, prediction_data=True)
                cluster_labels = clusterer.fit_predict(X_processed)

                # Points predicted as noise outliers are assigned the label -1, and this noise class needs to be removed as follows.
                M = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                _inner_print(args,f"[debug] hdbscan finds {M} clusters for class {c}", log_file_path)

                initial_centers = []
                initial_centers_map_original = {}
                initial_centers_map_path = {}
                # Use the real point with the highest probability as the initial center.
                _inner_print(args,f"[Class {c}] Atfer UMAP, find initial centers using max probability real points.", log_file_path)
                for cluster_id in range(M):
                    mask = (cluster_labels == cluster_id)
                    if not np.any(mask): continue
                    global_indices = np.where(mask)[0]
                    best_point_local_idx = np.argmax(clusterer.probabilities_[mask])
                    best_point_global_idx = global_indices[best_point_local_idx]
                    # a. Its UMAP coordinates will be passed to the post function as initial_centers
                    initial_centers.append(X_processed[best_point_global_idx])
                    # b. Its original high-dimensional coordinates will be saved for the final replacement
                    initial_centers_map_original[cluster_id] = X_original_unscaled[best_point_global_idx]
                    initial_centers_map_path[cluster_id] = original_paths[c][best_point_global_idx]

                ##########################################################################
                # Post-processing: Use three strategies to ensure the number of representative samples equals the IPC.
                ##########################################################################
                final_clusters_dict = hdbscan_post(args, M, initial_centers, cluster_labels, X_processed, log_file_path, clusterer)

                final_centers_original = []

                if final_clusters_dict:
                    for i, (cluster_id, cluster_info) in enumerate(final_clusters_dict.items()):
                        origin = cluster_info.get('origin', 'unknown')

                        if origin == 'hdbscan_initial':
                            final_center = initial_centers_map_original[cluster_id]
                            path_to_save = initial_centers_map_path[cluster_id]

                            mask = (cluster_labels == cluster_id)
                            global_indices = np.where(mask)[0]
                            best_point_local_idx = np.argmax(clusterer.probabilities_[mask])
                            center_local_idx = global_indices[best_point_local_idx]
                        else:
                            center_processed = cluster_info['center']
                            points_in_cluster_umap = X_processed[cluster_info['points_mask']]

                            # Find the closest point to the new cluster center
                            distances = euclidean_distances(points_in_cluster_umap, center_processed.reshape(1, -1))
                            closest_point_local_idx = np.argmin(distances)
                            global_indices_sub = np.where(cluster_info['points_mask'])[0]
                            center_local_idx = global_indices_sub[closest_point_local_idx]
                            final_center = X_original_unscaled[center_local_idx]
                            path_to_save = original_paths[c][center_local_idx]

                        final_centers_original.append(final_center)

                        ##########################################################################
                        # Save representative samples as set R.
                        ##########################################################################
                        try:
                            final_save_path = os.path.join(final_resized_real_image_dir, f"{i}.png")
                            image = Image.open(path_to_save).convert("RGB")
                            target_size = (args.size, args.size)
                            resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
                            resized_image.save(final_save_path)

                        except FileNotFoundError:
                            print(f"\nWarning: Original image not found at {path_to_save}. Skipping copy.")

                    clusters_centers[c] = np.array(final_centers_original)

            base_filename_without_ext, file_ext = os.path.splitext(args.saved_clusters_base_name)
            chunk_filename = f"{base_filename_without_ext}_{chunk_id}{file_ext}"
            chunk_save_path = os.path.join(args.specific_cluster_dir, chunk_filename)
            print(f"[Chunk Mode] Saving centers for chunk {chunk_id} to {chunk_save_path}")
            save_clusters(clusters_centers, chunk_save_path)

            del clusters_centers
            del original_features_per_class
            del original_paths
    # endregion

    # region Use set R to guide SDXL in generating images, obtaining the final set G.
    if args.generate_images:
        # Load and merge all the clusters_centers chunk
        clusters_centers = {}
        num_chunks = args.nclass // 10
        for chunk_id in range(num_chunks):

            base_filename_without_ext, file_ext = os.path.splitext(args.saved_clusters_base_name)
            chunk_filename = f"{base_filename_without_ext}_{chunk_id}{file_ext}"
            chunk_file_path = os.path.join(args.specific_cluster_dir, chunk_filename)

            if os.path.exists(chunk_file_path):
                print(f"Loading and merging: {chunk_file_path}")
                with open(chunk_file_path, "rb") as f:
                    chunk_data = pickle.load(f)
                clusters_centers.update(chunk_data)

            else:
                print(f"Warning: Chunk file not found, skipping: {chunk_file_path}")

        print(f"All chunks merged. Total classes loaded: {len(clusters_centers)}")

        args._class_labels = class_labels
        args._sel_classes = sel_classes
        args._class_id_to_name = class_id_to_name

        num_gpus = torch.cuda.device_count()
        args._num_gpus = num_gpus

        generate_images_multi_gpu(args, clusters_centers)
    # endregion

def get_args():
    parser = argparse.ArgumentParser(description="CoDA")

    parser.add_argument("--program_path", type=str, default='./', help='Base Dir')
    parser.add_argument('--dataset_dir', type=str, default='/root/autodl-tmp/datasets/ImageNet', help='ImageNet Dir')
    parser.add_argument('--local_model_path', type=str, default='/root/autodl-tmp/model/SDXL-Refiner', help='Model Dir')
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--phase", type=int, default=0)

    parser.add_argument("--spec",   type=str, default='none', help='Target Dataset')
    parser.add_argument("--nclass", type=int, default=10,     help='The class number of the target dataset')
    parser.add_argument("--IPC",    type=int, default=100,    help='Target IPC config')
    parser.add_argument("--size",   type=int, default=256,    help="Image size")

    # For get_features
    parser.add_argument("--calcu_features",      action="store_true", default=False, help="Calculate features for clustering")

    # For get clusters (representative samples, the set R)
    parser.add_argument("--calcu_cluster",       action="store_true", default=False, help="Perform clustering to obtain representative samples.")
    parser.add_argument("--begin_chunk_id",      type=int, default=0,  help="guidance opt steps")
    parser.add_argument("--n_neighbors",         type=int, default=15, help="umap n_neighbors")
    parser.add_argument("--min_samples",         type=int, default=3,  help="hdbscan min_samples")
    parser.add_argument("--min_cluster_size",    type=int, default=5,  help="hdbscan min_cluster_size")
    parser.add_argument("--num_seed_candidates", type=int, default=3,  help="Determine the number of candidate seed pairs.")

    parser.add_argument("--cluster_detial", action="store_true", default=False, help="whether to show the cluster details")
    parser.add_argument("--cluster_logger", action="store_true", default=False, help="whether to output the cluster details to txt")

    # For generate, the set G
    parser.add_argument("--generate_images",     action="store_true", default=False, help="Generate images.")
    parser.add_argument("--sample_step",         type=int,   default=25,  help="sample steps")
    # Base model denoising ratio. Actual steps = sample_step * denoising_factor. If < 1, the final steps are left for the Refiner.
    parser.add_argument("--denoising_factor",    type=float, default=1.0, help="Above")
    # The number of steps incorporating additional guidance during the base model's process, utilized to derive the Prior Injection Steps (PIS).
    # PIS = sample_step * denoising_factor * (1 - guideTPercent) + sample_step * (1-denoising_factor)
    parser.add_argument("--guideTPercent",       type=float, default=1.0, help="Above")
    parser.add_argument("--cfg_guidance_scale",  type=float, default=5.0, help="Standard cfg guidance scale")
    parser.add_argument("--CoDA_guidance_scale", type=float, default=0.1, help="CoDA guidance scale, AKA gamma")

    args = parser.parse_args()

    ############################################
    # postprocess the args
    ############################################
    args.specific_cluster_dir = os.path.join(args.program_path, "results/clusterfile", args.spec)
    args.features_cache_path = os.path.join(args.specific_cluster_dir, "original_features_cache.pkl")
    args.plot_dir = os.path.join(args.specific_cluster_dir, f"Minsize_{args.min_cluster_size}_n_neighbors_{args.n_neighbors}")
    args.log_file_path = os.path.join(args.plot_dir, "logs_cluster_details", f"IPC{args.IPC}", f"n_{args.n_neighbors}_s_{args.min_cluster_size}.txt")

    args.saved_clusters_base_name = f"{args.IPC}_n_{args.n_neighbors}_s_{args.min_cluster_size}_saved_clusters.pkl"

    _save_dir = os.path.join(args.program_path, "results", args.spec)
    args.save_dir = os.path.join(_save_dir, f"Step-{args.sample_step}/IPC-{args.IPC}/DF-{args.denoising_factor}-GTP-{args.guideTPercent}-"
                                            f"gamma-{args.CoDA_guidance_scale}/n_{args.n_neighbors}_s_{args.min_cluster_size}")

    os.makedirs(args.specific_cluster_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    return args

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method('spawn', force=True)

    args = get_args()
    main(args)