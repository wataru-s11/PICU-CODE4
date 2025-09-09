# PICU AI Code

This project contains scripts for evaluating patient vitals during surgery.

## Configuration

Paths used by `main_surgery.py` and `vital_reader.py` can be configured with environment variables, command line options, or a `config.json` file placed in the project root.

Environment variables take priority over values in `config.json`.

- `VITALS_PATH`: default path to a vitals CSV file.
- `BEDS_DIR`: directory containing per-bed files such as `vitals_history_2.csv`.
- `CVP_MODEL_PATH`: path to the CVP classification model (`.keras`).
- `SERVICE_ACCOUNT_FILE`: path to the Google Cloud service account JSON.
- `IMAGE_FOLDER`: directory containing monitor screenshots for `vital_reader.py`.

### Example `config.json`

Copy `config.example.json` to `config.json` and adjust the paths:

```json
{
  "VITALS_PATH": "/path/to/vitals_history.csv",
  "BEDS_DIR": "/path/to/beds_directory",
  "CVP_MODEL_PATH": "/path/to/cvp_model.keras",
  "SERVICE_ACCOUNT_FILE": "/path/to/service_account.json",
  "IMAGE_FOLDER": "/path/to/images"
}
```

### Environment variables

You may also set the environment variables instead of using `config.json`:

```bash
export VITALS_PATH=/path/to/vitals_history.csv
export BEDS_DIR=/path/to/beds_directory
export CVP_MODEL_PATH=/path/to/cvp_model.keras
export SERVICE_ACCOUNT_FILE=/path/to/service_account.json
export IMAGE_FOLDER=/path/to/images
```

Command line options override both environment variables and the configuration file. For example:

```bash
python vital_reader.py --cvp-model /path/to/model.keras --service-account-file /path/to/account.json --image-folder /path/to/images
```

Both methods allow the scripts to run on different operating systems without modifying the source code.

