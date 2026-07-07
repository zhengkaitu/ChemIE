# Object detection with RF-DETR

## Training and benchmarking (requiring Conda)

### 1. Create the Conda environment

```shell
$ conda create -y -n rf-detr -c conda-forge python=3.13.13 pip=26.0.1
$ conda activate rf-detr
$ pip install -r requirements.txt
```

The following steps assume that the `rf-detr` environment has been activated.

### 2. Prepare the data

Put `markush_annotations/` under `../data`, then run

```shell
$ python split_data.py
```

### 3. Train RF-DETR

```shell
$ python train_detr.py
```

### 4. Predict with trained model

```shell
$ predict_detr.py
```
