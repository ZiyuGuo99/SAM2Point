DATASET="Objaverse"   # Dataset name
TYPE="point"          # Prompt type - please choose from {'point', 'box', 'mask'}
SAMPLE_IDX=0          # Sample ID of the given 3D examples
PROMPT_IDX=0          # Prompt ID of the given prompt examples

python3 main.py --prompt_type $TYPE --sample_idx $SAMPLE_IDX --prompt_idx $PROMPT_IDX --dataset $DATASET
python3 gen_video.py --prompt_type $TYPE --sample_idx $SAMPLE_IDX --prompt_idx $PROMPT_IDX --dataset $DATASET
