# PICU AI Code

This project contains scripts for evaluating patient vitals during surgery.

## Installation

Install the optional screen-capture dependencies before running `auto_capture.py`:

```bash
pip install mss pillow
```

Install the OCR dependencies before running `vital_reader.py`:

```bash
pip install easyocr opencv-python
```

EasyOCR may also download PyTorch the first time it runs. GPU acceleration is
optional; the script will fall back to CPU-based OCR when PyTorch is not
available.

On Windows you can use:

```powershell
py -m pip install mss pillow
```

## Configuration

Paths used by `main_surgery.py` and `vital_reader.py` can be configured with environment variables, command line options, or a `config.json` file placed in the project root.

Environment variables take priority over values in `config.json`.

- `VITALS_PATH`: default path to a vitals CSV file.
- `BEDS_DIR`: directory containing per-bed files such as `vitals_history_2.csv`.
- `SERVICE_ACCOUNT_FILE`: path to the Google Cloud service account JSON.
- `IMAGE_FOLDER`: directory containing monitor screenshots for `vital_reader.py`.

### Example `config.json`

Copy `config.example.json` to `config.json` and adjust the paths:

```json
{
  "VITALS_PATH": "/path/to/vitals_history.csv",
  "BEDS_DIR": "/path/to/beds_directory",
  "SERVICE_ACCOUNT_FILE": "/path/to/service_account.json",
  "IMAGE_FOLDER": "/path/to/images"
}
```

### Environment variables

You may also set the environment variables instead of using `config.json`:

```bash
export VITALS_PATH=/path/to/vitals_history.csv
export BEDS_DIR=/path/to/beds_directory
export SERVICE_ACCOUNT_FILE=/path/to/service_account.json
export IMAGE_FOLDER=/path/to/images
```

Command line options override both environment variables and the configuration file. For example:

```bash
python vital_reader.py --spont-breath-model /path/to/model.pt --image-folder /path/to/images
```

Both methods allow the scripts to run on different operating systems without modifying the source code.

