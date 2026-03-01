# ChemIE
Chemical information extraction

## Deployment (requiring Docker)

### 1. Build and start the containerized service for chemical structure recognition using MolScribe.

```shell
$ make build-molscribe-image
$ make start-molscribe-service
````

The service can be stopped when no longer needed via

```shell
$ make stop-molscribe-service
```

### 3. Query the recognition service
We provide sample query commands in `scripts/query.sh`, which can be executed from the command line

```shell
$ bash scripts/query.sh
```

The responses from the service will be printed to the terminal, e.g.,
```shell
{
  "molblock": str
}
```

## Training and benchmarking (requiring Conda)

### 1. Create the Conda environment

```shell
$ conda create -y -n molscribe -c conda-forge python=3.9 ipykernel jupyterlab=4.0.13 packaging=21.3
$ conda activate molscribe
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```

The following steps assume that the `molscribe` environment has been activated.

### 2. Prepare the data

```shell
$ bash scripts/download_molscribe_data.sh
```

### 3. Download pretrained MolScribe checkpoint

```shell
$ bash scripts/download_pretrained_checkpoint.sh
```

### 4. Train MolScribe

```shell
$ bash scripts/train_molscribe.sh
```

### 5. Predict and evaluate with MolScribe

```shell
$ bash scripts/predict_molscribe.sh
$ bash scripts/evaluate_molscribe.sh
```
