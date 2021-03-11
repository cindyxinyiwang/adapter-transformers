#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
GPU=${1:-0}
MODEL=${2:-bert-base-multilingual-cased}
#MODEL=${2:-xlm-roberta-base}
DATA_DIR=${3:-"/home/xinyiw/download/"}
OUT_DIR=${4:-"$REPO/output/"}

export CUDA_VISIBLE_DEVICES=$GPU
TASK='panx'
LANGS="is"
TRAIN_LANGS="is"

NUM_EPOCHS=100
MAX_LENGTH=128
LR=1e-4
BPE_DROP=0

#LANG_ADAPTER="output/bert_s10k_gmask_is2no0.1_mlm/checkpoint-4000/is2fo/"
LANG_ADAPTER="is/wiki@ukp"

LANG_ADAPTER_NAME="is"
TASK_ADAPTER_NAME="en_ner"

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
for SEED in 3;
do
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}_${TASK_ADAPTER_NAME}_${LANG_ADAPTER_NAME}_bped${BPE_DROP}_s${SEED}/"

#TASK_ADAPTER="output/panx/bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangis_is_pretrainadapter_bped0_s${SEED}/checkpoint-best/is_ner/"

#TASK_ADAPTER="output/panx/bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangen_de_pretrained_bped0_s${SEED}/checkpoint-best/de_ner/"
#TASK_ADAPTER="output/panx/bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangen_en_pretrained_bped0_s${SEED}/checkpoint-best/en_ner/"

TASK_ADAPTER="output/panx/bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangen_en,de_pretrained_kl0.2_bped0_s${SEED}/checkpoint-best/ner/"
mkdir -p $OUTPUT_DIR
#  --init_checkpoint $MODEL_PATH \
python third_party/run_tag.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --seed $SEED \
  --learning_rate $LR \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --bpe_dropout $BPE_DROP \
  --test_adapter \
  --adapter_config pfeiffer \
  --task_name $TASK_ADAPTER_NAME \
  --predict_task_adapter $TASK_ADAPTER \
  --language "de" \
  --lang_adapter_config pfeiffer \
  --save_only_best_checkpoint $LC \
  --predict_lang_adapter  $LANG_ADAPTER
  #--predict_ensemble_lang_adapter "en/wiki@ukp"
  #--predict_lang_adapter $LANG_ADAPTER  \
  #--load_lang_adapter "is/wiki@ukp" 
done
