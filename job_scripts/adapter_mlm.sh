
export CUDA_VISIBLE_DEVICES=1

TRAIN_FILE=data/mono/is/iswiki.train.txt
META_TRAIN_FILE=data/mono/fo/fo0.1k.train.txt
TEST_FILE=data/mono/fo/fowiki.valid.txt
OUT_DIR=output/bert_gradmjoint_is2fo0.1k_mlm/
mkdir -p output/
mkdir -p $OUT_DIR

#    --mlm_augment 0.2 \
python third_party/run_language_modeling.py \
    --overwrite_output_dir \
    --output_dir=$OUT_DIR \
    --log_file=$OUT_DIR/train.log \
    --model_type=bert \
    --grad_mask \
    --model_name_or_path=bert-base-multilingual-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --meta_train_data_file=$META_TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --language is2fo \
    --train_adapter \
    --load_adapter "is/wiki@ukp" \
    --max_steps 1000 \
    --save_total_limit 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --adapter_config pfeiffer
