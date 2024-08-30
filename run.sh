DATASET="Objaverse"
TYPE="point"
SAMPLE_IDX=3
PROMPT_IDX=0

python3 main.py --prompt_type $TYPE --sample_idx $SAMPLE_IDX --prompt_idx $PROMPT_IDX --dataset $DATASET
python3 gen_video.py --prompt_type $TYPE --sample_idx $SAMPLE_IDX --prompt_idx $PROMPT_IDX --dataset $DATASET
