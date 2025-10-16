python eval_squadv2_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --limit 0


python eval_techqa_vllm.py \
  --model /home/brachmat/phd/models/Qwen2.5-7B-Instruct \
  --split validation \
  --batch_size  8 \
  --bnb8 \
  --limit 10

python augmentation_pubmed.py \
  --dataset pubmed \
  --data_path ../datasets/pubmed_final/train.csv \
  --flush_every 50 \
  --fsync \
  --n_calls_per_type 2 \
  --limit 0 \
  --out_jsonl augmented_pubmed_train.jsonl \
  --out_csv augmented_pubmed_train.csv


python augmentation_squad.py \
  --dataset squadqa \
  --data_path ../datasets/squad_v2_final/train.csv \
  --flush_every 50 \
  --fsync \
  --n_calls_per_type 2 \
  --limit 0 \
  --out_jsonl augmented_squad_train.jsonl \
  --out_csv augmented_squad_train.csv


i have these files and each they have experiments, N is the number of sample and baseline, etc are the type of experiments, and there are qwen and llama. I want to use with unsloth 
i need you to access datasets/pubmed_final/test.csv and datasets/squad_v2/test.csv i want to only save generated_answer and its id. so in total i will have 20 files for each of the experiments. 
I will also have to run the vanilla as well     "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
SYS_PROMPT = (
    "You are a careful assistant for extractive question answering. "
    "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
)
USER_TEMPLATE = (
    "Answer the question strictly based on the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
runs_unsloth_dpp/
20251008_181733_qwen_baseline_N6750_gpus2
20251008_190339_qwen_semantic_N6750_gpus2
20251008_195224_qwen_syntactic_N6750_gpus2
20251008_209541_qwen_lexical_N6750_gpus2
20251008_212719_qwen_all_N6750_gpus2
20251009_020031_qwen_baseline_N13500_gpus2
20251009_020130_qwen_semantic_N13500_gpus2
20251009_034202_qwen_syntactic_N13500_gpus2
20251009_051826_qwen_lexical_N13500_gpus2
20251009_065159_qwen_all_N13500_gpus2
20251009_123059_llama_baseline_N6750_gpus2
20251009_134922_llama_semantic_N6750_gpus2
20251009_145052_llama_syntactic_N6750_gpus2
20251009_153120_llama_lexical_N6750_gpus2
20251009_162316_llama_all_N6750_gpus2
20251009_192950_llama_baseline_N13500_gpus2
20251009_211713_llama_semantic_N13500_gpus2
20251009_230226_llama_syntactic_N13500_gpus2
20251010_004609_llama_lexical_N13500_gpus2
20251010_023001_llama_all_N13500_gpus2

these are pubmed runs
runs_unsloth_dpp_pubmed/
20251010_143412_qwen_baseline_N11250_gpus2
20251010_163436_qwen_semantic_N11250_gpus2
20251010_183108_qwen_syntactic_N11250_gpus2
20251010_202548_qwen_lexical_N11250_gpus2
20251010_222003_qwen_all_N11250_gpus2
20251011_052933_qwen_baseline_N22500_gpus2
20251011_092817_qwen_semantic_N22500_gpus2
20251011_131937_qwen_syntactic_N22500_gpus2
20251011_170552_qwen_lexical_N22500_gpus2
20251011_205216_qwen_all_N22500_gpus2
20251012_111911_llama_baseline_N11250_gpus2
20251012_132515_llama_semantic_N11250_gpus2
20251012_152359_llama_syntactic_N11250_gpus2
20251012_172010_llama_lexical_N11250_gpus2
20251012_191717_llama_all_N11250_gpus2
20251013_024036_llama_baseline_N22500_gpus2
20251013_064852_llama_semantic_N22500_gpus2
20251013_104419_llama_syntactic_N22500_gpus2


python eval.py --dataset squad --limit 200 --models qwen --include-vanilla --run-pattern baseline