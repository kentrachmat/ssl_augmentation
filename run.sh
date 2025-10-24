export RUN_ID=$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1 

# for model in qwen llama; do
#   for N in 8125 ; do
#     for cfg in baseline; do
#       export RUN_ID=$(date +%Y%m%d_%H%M%S)
#       echo ">>> Running $model | N=$N | cfg=$cfg"
#       torchrun --nproc_per_node=2 pararel_finetuning_clean.py -N $N --dataset squad --model $model --cfg $cfg
#     done
#   done
# done

for model in qwen ; do
  for N in 13187 ; do
    for cfg in baseline; do
      export RUN_ID=$(date +%Y%m%d_%H%M%S)
      echo ">>> Running $model"
      torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset pubmed --model $model -N $N --cfg $cfg 
    done
  done
done

python lora_eval.py --dataset squad_v2
python lora_eval.py --dataset pubmed
