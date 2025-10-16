export RUN_ID=$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# torchrun --nproc_per_node=2 pararel_finetuning.py -N 300 --model qwen --cfg semantic
# torchrun --nproc_per_node=2 pararel_finetuning.py -N 300 --model qwen --cfg syntactic
# torchrun --nproc_per_node=2 pararel_finetuning.py -N 300 --model qwen --cfg lexical
# torchrun --nproc_per_node=2 pararel_finetuning.py -N 300 --model qwen --cfg all
# (54000, 27000.0, 13500.0, 6750.0)

# for model in qwen llama; do
#   for N in 11250 22500 ; do
#     for cfg in baseline semantic syntactic lexical all; do
#       export RUN_ID=$(date +%Y%m%d_%H%M%S)
#       echo ">>> Running $model | N=$N | cfg=$cfg"
#       torchrun --nproc_per_node=2 pararel_finetuning.py -N $N --model $model --cfg $cfg
#     done
#   done
# done


for model in qwen llama; do
  for dataset in squad pubmed; do
    export RUN_ID=$(date +%Y%m%d_%H%M%S)
    echo ">>> Running $model"
    torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset $dataset --model $model  
  done
done