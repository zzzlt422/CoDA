#nvidia-smi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

IMAGENET_FOLDER="/root/autodl-tmp/datasets/ImageNet"
MODEL_FOLDER="/root/autodl-tmp/model/SDXL-Refiner"

run_experiment() {
    local run_step1=${1:-true}
    local flag_features=${2:-false}
    local flag_cluster=${3:-false}
    local flag_generate=${4:-false}
    local run_step2=${5:-true}

    local run_stages=""
    if [[ "$flag_features" == "true" ]]; then
        run_stages="$run_stages --calcu_features"
    fi
    if [[ "$flag_cluster" == "true" ]]; then
        run_stages="$run_stages --calcu_cluster"
    fi
    if [[ "$flag_generate" == "true" ]]; then
        run_stages="$run_stages --generate_images"
    fi

    if [[ "$run_step1" == "true" ]]; then

        python CoDA_main.py \
            --dataset_dir "$IMAGENET_FOLDER" --local_model_path "$MODEL_FOLDER" \
            --spec "$SPEC" \
            --IPC "$ipc" \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" \
            --cluster_detial --cluster_logger \
            --sample_step "$timestep" --denoising_factor "$DF" --guideTPercent "$GTP" --CoDA_guidance_scale "$gamma" \
            $run_stages

    fi

    if [[ "$run_step2" == "true" ]]; then

        local train_data_path="./results/${SPEC}/Step-${timestep}/IPC-${ipc}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${n_neighbors}_s_${size_min}"
        local val_data_path="$IMAGENET_FOLDER/validation"

        local use_real_images=${6:-true}
        if [[ "$use_real_images" == "true" ]]; then
            train_data_path+="/real_images"
        else
            train_data_path+="/generated_images"
        fi

        local train_save_dir="./trained_results/ipc${ipc}/n_${n_neighbors}_s_${size_min}/step-$timestep-DF-$DF/GTP-$GTP-gamma-$gamma"

        echo "==> Testing with ResNet-AP 10..."
        python ./test/train.py --dataset_dir "$train_data_path" "$val_data_path" \
            -d imagenet --spec "$SPEC" --nclass 10  --size 256 --ipc "$ipc" \
            -n resnet_ap --depth 10  --save-dir "$train_save_dir-resnet_ap"  \
            --workers 12 \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" --tag test
    fi
}

export CUDA_VISIBLE_DEVICES=0,1,2,3

ipc=10

n_neighbors=85
size_min=55

timestep=25
DF=1.0
GTP=0.9
gamma=0.05

#SPEC_LIST="imageA imageB imageC imageD imageE IDC nette"
SPEC_LIST="imageA"
for SPEC in $SPEC_LIST; do
    #                Step1  cal_features cal_cluster generate   Step2   use_real_images
    run_experiment   true       true        true       true     true         true
done

# cd /root/autodl-tmp/CoDA
# conda activate MG
# scripts/CoDA.sh