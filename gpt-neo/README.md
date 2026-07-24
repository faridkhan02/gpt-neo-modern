# GPT-Neo TPU Training Project

## Overview

This repository contains a GPT-style language model training project built on TensorFlow, Mesh TensorFlow, and a custom GPT-like pipeline.

### Contact

- LinkedIn: https://www.linkedin.com/in/md-farid-khan-bb662b2b4/

This code is intended for local model training and generation. It is not an OpenAI or Hugging Face hosted API client.

This project is a backend machine learning pipeline only. It does not include a web-based frontend or UI application.

## Key Points

- No external API key is required to run this repository.
- The model and training settings are configured with JSON files in `configs/`.
- The main entrypoint is `main.py`.
- `run_experiment.py` is an optional TPU/MongoDB orchestration wrapper.

## Recommended Setup

- Python 3.8 or 3.9
- Windows, Linux, or macOS
- A virtual environment for package isolation

## Install Dependencies

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> If `tensorflow==2.5.1` fails to install, switch to Python 3.8 or 3.9 and retry.

## Project Structure

```text
c:\python-beginner-projects\got-neo
|-- configs/
|   |-- gpt2_small.json
|   |-- gpt3_13B_256.json
|   `-- dataset_configs/
|-- data/
|   |-- create_tfrecords.py
|   |-- encoders.py
|   `-- train_tokenizer.py
|-- models/
|   |-- activations.py
|   |-- layers.py
|   `-- gpt2/gpt2.py
|-- main.py
|-- run_experiment.py
|-- requirements.txt
|-- inputs.py
|-- export.py
|-- model_fns.py
`-- README.md
```

## How to Run

### Show help for the training script

```powershell
python main.py --help
```

### Train the model

```powershell
python main.py --tpu device:GPU:0 --model configs/gpt2_small.json --steps_per_checkpoint 5000
```

> Note: the default production model config uses Google Cloud Storage paths and requires local checkpoint/dataset files or GCS access.

### Local runnable demo

A lightweight local demo configuration is included in `configs/synthetic_demo.json`. It does not require any external dataset or model checkpoint.

```powershell
python main.py --model configs/synthetic_demo.json --check_dataset
```

### Local web frontend demo

A simple Flask frontend is included in `app.py` to submit prompts to the local backend. It uses `main.py` with `configs/synthetic_demo.json`.

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

If `conda` is not on your PowerShell PATH, run the environment Python directly:

```powershell
powershell -NoProfile -Command "& 'C:\Users\khana\anaconda3\envs\gotneo39\python.exe' main.py --model configs/synthetic_demo.json --check_dataset"
```

The synthetic demo prints raw token IDs instead of decoded text because it does not use a real encoder or dataset.

### Generate text

1. Create a prompt file named `prompt.txt`:

```text
Artificial intelligence is transforming the way we
```

2. Run prediction:

```powershell
python main.py --tpu device:GPU:0 --model configs/gpt2_small.json --predict --prompt prompt.txt
```

### Optional Sacred/MongoDB experiment launcher

```powershell
python run_experiment.py --tpu <TPU_NAME> --model configs/gpt2_small.json --experiment_name my_run
```

## Configuration Notes

- `configs.py` loads model parameter files from `configs/`.
- Each model config must reference a dataset from `configs/dataset_configs/`.
- `main.py` uses `tensorflow.compat.v1` and `mesh_tensorflow`.

## Known Issues and Exact Errors

- `ModuleNotFoundError: No module named 'mesh_tensorflow'` means the dependency is not installed. Run `python -m pip install -r requirements.txt`.
- `ImportError` for TensorFlow may occur if you are using Python 3.10+ with `tensorflow==2.5.1`. Use Python 3.8 or 3.9.
- `UnimplementedError: File system scheme 'gs' not implemented` occurs when the model or dataset config points to Google Cloud Storage paths like `gs://neo-models/...`. This repository currently needs local checkpoint and dataset files or a TensorFlow build with GCS support.
- `ValueError: tf.enable_eager_execution must be called at program startup.` was previously triggered by the `--check_dataset` path; it has been fixed by switching `check_dataset()` to TensorFlow v1 graph mode.
- `main.py` can run with `--model configs/synthetic_demo.json` for a local validation, but the full production model configs still require accessible datasets and checkpoint paths.
- If `conda` is not available on PowerShell PATH, use the direct interpreter at `C:\Users\khana\anaconda3\envs\gotneo39\python.exe`.
- `run_experiment.py` requires MongoDB at `127.0.0.1:27017` with credentials `user` / `password` as currently configured.
- `run_experiment.py` also assumes the `pu` command is available for TPU recreation.

## Dependencies

This repository depends on the packages listed in `requirements.txt`, including:

- `tensorflow==2.5.1`
- `mesh_tensorflow==0.1.18`
- `tensorflow-datasets==3.2.1`
- `tokenizers==0.9.4`
- `transformers==4.1.1`
- `sacred`

## No API Key Required

This repository runs locally and does not require any external API key.

## Troubleshooting

- If `python main.py --help` fails with missing modules, install the packages from `requirements.txt`.
- If TensorFlow or Mesh TensorFlow fails to install, try Python 3.8 or 3.9.
- If `run_experiment.py` is used, ensure MongoDB is running and accessible at `127.0.0.1:27017`.
- If `main.py` returns a `ValueError` or configuration error, verify the selected model config file path and dataset settings.
