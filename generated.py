import os
import torch
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing as mp

from Loadmodel import load_sdxl_and_refiner

import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def process_and_save_image(image, target_size, save_path):
    """
    Resize and save images in a background thread.
    """
    try:
        if image.size == target_size:
            target_size_image = image
        else:
            target_size_image = image.resize(target_size, Image.Resampling.LANCZOS)

        target_size_image.save(save_path)
        # print(f"Image saved to {save_path}")

    except Exception as e:
        print(f"Error processing and saving image: {e}")


def generate_images_single_gpu(gpu_id, args, clusters_centers, my_assignments, results_dict):

    try:

        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        with suppress_stdout():
            pipeline, refiner = load_sdxl_and_refiner(args)
        pipeline.to(device)
        refiner.to(device)

        batch_size = 1
        class_labels = args._class_labels
        sel_classes = args._sel_classes
        class_id_to_name = args._class_id_to_name
        save_dir = os.path.join(args.save_dir, "generated_images")

        base_seed = args.seed + gpu_id * 10000

        total_tasks = sum(len(tasks) for tasks in my_assignments.values())
        progress_bar = tqdm(total=total_tasks, desc=f"GPU {gpu_id}", position=gpu_id)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for class_idx, (class_label, sel_class) in enumerate(zip(class_labels, sel_classes)):

                index = sel_classes.index(sel_class)
                whole_class_name = class_id_to_name[sel_class]
                first_class_name = whole_class_name.split(',')[0].strip()
                save_dir_class = os.path.join(save_dir, sel_class)
                os.makedirs(save_dir_class, exist_ok=True)

                assignments = my_assignments.get(class_idx, [])
                for shift in assignments:

                    image_seed = base_seed + class_idx * args.IPC + 1000 + shift
                    generator = torch.Generator(device=device).manual_seed(image_seed)

                    save_path = os.path.join(save_dir_class, f"{shift}.png")

                    target_size = (args.size, args.size)
                    if args.CoDA_guidance_scale > 0.0:
                        represent_latent = torch.tensor(clusters_centers[index][shift].reshape(1, 4, 128, 128))

                    with suppress_stdout():
                        with torch.no_grad():
                            ################################################################
                            # Use 1st official class descriptor only; no negative prompt.
                            ################################################################
                            negative_prompt = None
                            prompt = first_class_name

                            ################################################################
                            # When DF=1.0, use only the SDXL Base Pipeline for generation.
                            ################################################################
                            if args.denoising_factor == 1.0:

                                # Base generation
                                pipeline_kwargs = {
                                    "generator": generator,

                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt,

                                    "num_inference_steps": args.sample_step,
                                    "guideTPercent": args.guideTPercent,

                                    "guidance_scale": args.cfg_guidance_scale,
                                    "CoDA_guidance_scale": args.CoDA_guidance_scale,
                                }
                                if args.CoDA_guidance_scale > 0.0:
                                    pipeline_kwargs["represent_latent"] = represent_latent

                                pipeline_output, final_latent = pipeline(**pipeline_kwargs)

                                if torch.isnan(final_latent).any() or torch.isinf(final_latent).any():
                                    print(f"!!! WARNING: Final latent contains NaN or Inf after custom pipeline processing!")

                                image = pipeline_output.images[0]
                                executor.submit(process_and_save_image, image, target_size, save_path)

                            else:
                                # Base generation
                                pipeline_kwargs = {
                                    "output_type": "latent",
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt,
                                    "num_inference_steps": args.sample_step,
                                    "denoising_end": args.denoising_factor,
                                    "guideTPercent": args.guideTPercent,
                                    "guidance_scale": args.cfg_guidance_scale,
                                    "CoDA_guidance_scale": args.CoDA_guidance_scale,
                                    "generator": generator
                                }
                                if args.CoDA_guidance_scale > 0.0:
                                    pipeline_kwargs["represent_latent"] = represent_latent
                                pipeline_output, final_latent_from_base = pipeline(**pipeline_kwargs)
                                latent_image = pipeline_output.images

                                refiner_kwargs = {
                                    "image": latent_image,
                                    "prompt": prompt,
                                    "negative_prompt": negative_prompt,
                                    "num_inference_steps": args.sample_step,
                                    "denoising_start": args.denoising_factor,
                                    "generator": generator
                                }
                                # Refiner
                                refiner_output_obj, final_latent = refiner(**refiner_kwargs)

                                if torch.isnan(final_latent).any() or torch.isinf(final_latent).any():
                                    print(
                                    f"!!! WARNING: Final latent for plotting still contains NaN or Inf after custom pipeline processing!")

                                image = refiner_output_obj.images[0]
                                executor.submit(process_and_save_image, image, target_size, save_path)

                    progress_bar.update(1)
        progress_bar.close()
        print(f"GPU {gpu_id} completed all tasks")

    except Exception as e:
        print(f"Error in GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()


def generate_images_multi_gpu(args, clusters_centers):

    num_gpus = args._num_gpus
    class_labels = args._class_labels
    sel_classes = args._sel_classes
    num_samples_per_class = args.IPC

    ##############################################
    # Distribute generation tasks to each GPU.
    ##############################################
    gpu_assignments = {}
    for gpu_id in range(num_gpus):
        gpu_assignments[gpu_id] = {}

    extra_task_gpu_offset = 0
    for class_idx in range(len(class_labels)):
        images_per_gpu = num_samples_per_class // num_gpus
        extra_images = num_samples_per_class % num_gpus

        current_shift = 0
        for gpu_id in range(num_gpus):
            gpu_assignments[gpu_id][class_idx] = []

            for _ in range(images_per_gpu):
                gpu_assignments[gpu_id][class_idx].append(current_shift)
                current_shift += 1

        for i in range(extra_images):
            gpu_to_get_extra = (extra_task_gpu_offset + i) % num_gpus
            gpu_assignments[gpu_to_get_extra][class_idx].append(current_shift)
            current_shift += 1

        extra_task_gpu_offset = (extra_task_gpu_offset + extra_images) % num_gpus

    # print("GPU assignments:")
    # for gpu_id, assignments in gpu_assignments.items():
    #     print(f"GPU {gpu_id}: {assignments}")

    manager = mp.Manager()
    results_dict = manager.dict()

    for i in range(len(class_labels)):
        results_dict[i] = manager.list()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=generate_images_single_gpu,  # Launch generate_images_single_gpu on each GPU.
                       args=(gpu_id, args, clusters_centers, gpu_assignments[gpu_id], results_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # print("All GPU processes completed")
