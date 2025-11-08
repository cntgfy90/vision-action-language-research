# VLA reaserch

In current project we research on action output tokenizers that may be potentially incorporated as VLA action output layer.

### How to run locally

We provide 2 options to run the project: either on local machine directly or via docker.

#### Using local machine setup

**Note**: this option might not work if your local machine setup diverges from the ones we worked on. For reference we attach below the related versions of tools we used:

- Python: 3.10.18
- uv: 0.9.5

If all good, then from the project root run next commands:

1. Create virtual environment:

```bash
uv venv
```

2. Install the dependencies:

```bash
uv pip install -r requirements.txt
```

3. Enable local env:

```bash
source .venv/bin/activate
```

#### Using docker-compose

1. Run from project root (suggested with `--build` command to avoid any cache issues)

```bash
docker-compose up --build
```

**Note**: It may take up to 10 minutes to run with docker. We agree it is too long however we have not optimized booting pipeline yet.

2. Open in browser `http://localhost:8888/lab?`


### Project structure

**The main research was done in `main.ipynb`**: feel free to run cell by cell on observe the results obtained from various tokenizers.

In `src` we encapsulated all the necessary modules we rely on:
- `visualizations.py`: visualization utils
- `datasets.py`: normalized datasets
- `metrics`: contains the measure utils that is the main benchmark technique in the project
- `tokenizers`: tokenizer implementations based on base `Tokenizer` abstract class from `src/tokenizers/base.py`

### Formatting

We use `black` as a formatter. It should auto-fix if run with the command below:

```bash
black
```
