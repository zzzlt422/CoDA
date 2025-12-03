import torch
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast

from data import ImageFolder
from Loadmodel import load_sdxl_and_refiner

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_distributed_loader(args, rank, world_size, return_path=False, mode_id_file=None):
    transform = transforms.Compose([
        # transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset_train = os.path.join(args.dataset_dir, "train")
    dataset = ImageFolder(dataset_train, transform=transform, nclass=args.nclass,
                          spec=args.spec, phase=args.phase, seed=0,
                          return_origin=True, return_path=return_path, mode_id_file=mode_id_file)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    loader = DataLoader(dataset, batch_size=8, num_workers=4,
                        sampler=sampler, drop_last=False, pin_memory=True,)
    return loader

def merge_distributed_features(world_size, features_cache_path, args):
    print("Merging features from all processes...")
    nclass = args.nclass

    chunked_data = defaultdict(lambda: {"features": defaultdict(list), "paths": defaultdict(list)})

    for rank in range(world_size):
        rank_cache_path = f"{features_cache_path}.rank{rank}"
        if os.path.exists(rank_cache_path):
            with open(rank_cache_path, "rb") as f:
                cache_data = pickle.load(f)

            for label, features in cache_data["features"].items():
                chunk_id = label // 10
                chunked_data[chunk_id]["features"][label].extend(features)

            if "paths" in cache_data:
                for label, items in cache_data["paths"].items():
                    chunk_id = label // 10
                    chunked_data[chunk_id]["paths"][label].extend(items)

            os.remove(rank_cache_path)

    num_chunks = nclass // 10
    for chunk_id in range(num_chunks):
        if chunk_id in chunked_data:
            data_chunk = chunked_data[chunk_id]
            chunk_file_path = f"{features_cache_path}_{chunk_id}"

            with open(chunk_file_path, "wb") as f:
                pickle.dump(data_chunk, f)
            print(f"Saved chunk {chunk_id} to {chunk_file_path}")

    print(f"Successfully merged features into {num_chunks} chunks.")

def extract_features_distributed(rank, world_size, args):
    try:
        features_cache_path = args.features_cache_path
        setup_distributed(rank, world_size)
        torch.manual_seed(args.seed)
        device = torch.device(f'cuda:{rank}')

        vae16 = load_sdxl_and_refiner(args, VAE16_ONLY=True, VAEFIX=True)
        vae16 = vae16.to(device).eval()
        encoder=vae16
        print(f"Rank {rank}: VAE model loaded.")

        loader = get_distributed_loader(args, rank, world_size, return_path=True)

        features_per_class = defaultdict(list)
        path_per_class = defaultdict(list)
        index_per_class = defaultdict(list)

        print(f"Successfully! Rank {rank}: Starting feature extraction...")

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GPU {rank}", position=rank):
                images, labels, _, paths = batch
                images = images.to(device).half()

                with autocast():
                    features = encoder.encode(images).latent_dist.mean * encoder.config.scaling_factor

                if torch.isnan(features).any() or torch.isinf(features).any():
                    print(f"WARNING: Rank {rank} - Features contain NaN or Inf. Labels: {labels}")

                features = features.detach().cpu()
                batch_size = features.size(0)
                flattened_features = features.view(batch_size, -1)

                for i in range(batch_size):
                    feature = flattened_features[i]
                    label = labels[i].item()
                    path = paths[i]
                    path_per_class[label].append(path)
                    features_per_class[label].append(feature)

        rank_cache_path = f"{features_cache_path}.rank{rank}"
        with open(rank_cache_path, "wb") as f:
            pickle.dump({"features": features_per_class, "paths": path_per_class}, f)

        print(f"Rank {rank}: Saved features to {rank_cache_path}")

        dist.barrier(device_ids=[rank])

        if rank == 0:
            merge_distributed_features(world_size, features_cache_path, args)

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise e
    finally:
        dist.destroy_process_group()

def calculate_features_multiprocess(args):
    world_size = torch.cuda.device_count()
    print(f"Starting multi-process distributed feature extraction with {world_size} GPUs")

    mp.spawn(
        extract_features_distributed,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )