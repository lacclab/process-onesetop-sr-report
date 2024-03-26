
# OneStopGaze-Preprocessing

TODO update 


All scripts related to preprocessing OneStopGaze

Curently includes preprocessing the interest_area report.

See [here](data_notes.md) for more information about the data.

## Installation

For development, clone the repository and install the package in editable mode:

NOTE: You need to have access to the repo. 
profile -> settings -> developer settings -> personal access token -> classic -> generate new token (if existing is expired) -> give permissions
-> Use instead of github password when trying to clone , fetch. etc.


```bash
git clone
cd OneStopGaze-Preprocessing

conda env create -f env.yaml
```

For production, install the package from GitHub:

```bash
pip install git+https://github.com/lacclab/onestopgaze-preprocessing.git
```

## Usage

To parse the data from inside a script:

```python
import onestopgaze_preprocessing.preprocessing as prp
cfg = prp.argsParser().parse_args()  
prp.preprocess_data(prp.cfg)
```

To parse the data from the command line:  

```bash
python -m onestopgaze_preprocessing.preprocessing --ROOT_DIR <path_to_root_data_folder>
```

Assumes the raw tsv files to be placed in `hunting_data_path` and `gathering_data_path`, and will be saved to 
`save_path`. 
To see all the options:


```bash
python onestopgaze_preprocessing/preprocessing.py -h
```


Example script:
```python
""" This script is used to parse the raw data and save it in a format 
that is easier to work with. """
from pathlib import Path

import onestopgaze_preprocessing.preprocessing as prp


def process_data(mode: str):
    data_dir = Path(__file__).absolute().parent.parent / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"

    hunting_file = f"n_{mode}_report_all_variables.tsv"
    gathering_file = f"p_{mode}_report_all_variables.tsv"
    save_file = f"{mode}_data_enriched_350.csv"
    args_file = f"{mode}_preprocessing_args_350.json"

    args = [
        "--hunting_data_path",
        str(raw_dir / hunting_file),
        "--gathering_data_path",
        str(raw_dir / gathering_file),
        "--save_path",
        str(interim_dir / save_file),
        "--mode",
        mode,
        "--filter_query",
        "practice==0 & reread==0",
    ]

    cfg = prp.ArgsParser().parse_args(args)

    args_save_path = interim_dir / args_file
    cfg.save(str(args_save_path))
    print(f"Saved config to {args_save_path}")

    print(f"Running preprocessing with args: {args}")
    prp.preprocess_data(cfg)


if __name__ == "__main__":
    process_data(prp.Mode.FIXATION.value)
    process_data(prp.Mode.IA.value)
```
