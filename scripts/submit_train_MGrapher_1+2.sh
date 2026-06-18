IFS=","

export BATCH_SIZE=4
export LR=2e-4
export EPOCH=20
export EXP_NO=MGrapher_1+2

mkdir -p logs/$EXP_NO

export SAVE_PATH=output/${EXP_NO}
# Order matters: dataset dir i must align with train CSV i.
# Val uses the first dataset dir (matches VAL_FILE).
export DATASET_DIR_TRAIN=data/MG1/markushgrapher-synthetic-training,data/MG2/uspto-mol-m-54k
export TRAIN_FILE=data/MG1/markushgrapher-synthetic-training/synthetic-train.processed.csv,data/MG2/uspto-mol-m-54k/uspto-mol-m-54k-train.processed.csv
export VAL_FILE=data/MG1/markushgrapher-synthetic-training/synthetic-test.processed.csv
sh scripts/train.sh > logs/$EXP_NO/$EXP_NO.log
