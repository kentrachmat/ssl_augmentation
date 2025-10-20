export RUN_ID=$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1 

# for model in qwen llama; do
#   for N in 11250 22500 ; do
#     for cfg in baseline semantic syntactic lexical all; do
#       export RUN_ID=$(date +%Y%m%d_%H%M%S)
#       echo ">>> Running $model | N=$N | cfg=$cfg"
#       torchrun --nproc_per_node=2 pararel_finetuning.py -N $N --model $model --cfg $cfg
#     done
#   done
# done

# 6593
for model in llama; do
  for N in 1648 3296 ; do
    for cfg in baseline semantic syntactic lexical all; do
      export RUN_ID=$(date +%Y%m%d_%H%M%S)
      echo ">>> Running $model"
      torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset pubmed --model $model -N $N --cfg $cfg 
    done
  done
done

# 4062
for model in qwen llama; do
  for N in 1015 2031 ; do
    for cfg in baseline semantic syntactic lexical all; do
      export RUN_ID=$(date +%Y%m%d_%H%M%S)
      echo ">>> Running $model"
      torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset squad --model $model -N $N --cfg $cfg 
    done
  done
done

python lora_eval.py --dataset pubmed
python lora_eval.py --dataset squad_v2
python em_f1.py

for model in qwen llama; do
  for N in 6593 ; do
    for cfg in baseline semantic syntactic lexical all; do
      export RUN_ID=$(date +%Y%m%d_%H%M%S)
      echo ">>> Running $model"
      torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset pubmed --model $model -N $N --cfg $cfg 
    done
  done
done

for model in qwen llama; do
  for N in 4062 ; do
    for cfg in baseline semantic syntactic lexical all; do
      export RUN_ID=$(date +%Y%m%d_%H%M%S)
      echo ">>> Running $model"
      torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset squad --model $model -N $N --cfg $cfg 
    done
  done
done



# export RUN_ID=$(date +%Y%m%d_%H%M%S)
# torchrun --nproc_per_node=2 pararel_finetuning_clean.py --dataset squad --model qwen -N 906 --cfg all 