
export CUDA_VISIBLE_DEVICES=1

TRAIN_FILE=data/mono/is/iswiki.train.txt
#META_TRAIN_FILE=data/mono/no/no0.1k.train.txt
TEST_FILE=data/mono/fo/fowiki.valid.txt
#OUT_DIR=output/bert_metaaug0.0001_is2no0.1_mlm/
OUT_DIR=output/bert_is0.2_mlm/
mkdir -p output/
mkdir -p $OUT_DIR

#    --mixup_tau 0.5 \
#    --grad_mask \
#    --meta_augment_w 0.0001 \
#    --meta_train_data_file=$META_TRAIN_FILE \
python third_party/run_language_modeling.py \
    --mlm_augment 0.2 \
    --block_size 256 \
    --overwrite_output_dir \
    --output_dir=$OUT_DIR \
    --log_file=$OUT_DIR/train.log \
    --model_type=bert \
    --model_name_or_path=bert-base-multilingual-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --language is2fo \
    --train_adapter \
    --load_adapter "is/wiki@ukp" \
    --max_steps 1000 \
    --save_total_limit 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --adapter_config pfeiffer
