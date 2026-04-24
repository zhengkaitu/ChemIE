IFS=","

export BATCH_SIZE=8
export LR=4e-4
export EPOCH=20
export EXP_NO=MGrapher

mkdir -p logs/$EXP_NO

export SAVE_PATH=output/${EXP_NO}
export DATASET_DIR_TRAIN=data/hf/markushgrapher-synthetic-training
export TRAIN_FILE=data/hf/synthetic-train.processed.csv
export VAL_FILE=data/hf/synthetic-val.processed.csv
sh scripts/train.sh > logs/$EXP_NO/$EXP_NO.log
