# Lab 1 - git/dvc and data preparation

## What was done

| Step | Where |
|---|---|
| uv project | `pyproject.toml`, `uv.lock`, `.python-version` (commit `91a1b1d`) |
| dvc initialised | `.dvc/`, `.dvcignore` (commit `e67fbba`) |
| DagsHub as the dvc remote | `.dvc/config` (commit `7c4009c`, URL added to the project config later) |
| Raw data tracked | `data.dvc`, `/data` in `.gitignore` (commit `2625e5e`) |
| ImageFolder datasets (128x128, one folder per class, full + mini) | [`src/food11/data.py`](../src/food11/data.py) |

**Data option adopted:** none of the fallback options was needed. The full raw dataset
(16,643 images, about 1.19 GB) was pushed to the DagsHub remote
`https://dagshub.com/urpx-prog/mlops-lab-1.dvc`.

### The data preparation script

```bash
uv run python ./src/food11/data.py
```

`data/food11_raw` keeps the Kaggle layout: three flat folders, with the class id at the
start of each file name (`3_127.jpg` is an Egg image). `data.py` reads every image,
center-crops it to a square and shrinks it to 128x128 (`ImageOps.fit`, so the image is not
stretched). It then saves it into a folder named after its class:

```
data/food11_processed/{training,validation,evaluation}/<class>/*.jpg       9866 / 3430 / 3347 images
data/food11_processed_mini/{training,validation,evaluation}/<class>/*.jpg  at most 100 per class
```

This is the layout ResNet training expects through `torchvision.datasets.ImageFolder`.
Each sub-folder is a class, and the folder name becomes the label. The mini dataset is a
fixed random sample (seed 42) of the processed images, at most 100 per class and split.
It is used to check that the training code works before spending time on the full set.
Resizing runs on all CPU cores, and the script deletes and rebuilds both folders on every
run, so it is safe to run again.

---

## Answers

### Question 1 - Files created by `uv init`

- `pyproject.toml` describes the project: name, version, description, required Python
  version (`>=3.12`) and the `dependencies` list, which starts empty. `uv add` writes
  into it.
- `.python-version` holds the Python version uv uses for this project (`3.12`).
- `main.py` is a "Hello from mlops-lab-1!" starter script.
- `README.md` is an empty project readme.
- `.gitignore` already ignores `.venv/`, `__pycache__/` and compiled Python files.

`uv.lock` was created the first time uv resolved the environment (`uv run` / `uv sync`).
It pins the exact version, source and hashes of every package, including transitive
ones, so `uv sync` rebuilds the same environment on any machine. `pyproject.toml` says
what we want; `uv.lock` says exactly what was installed. Both are committed. The `.venv/`
folder is the local virtual environment and is not committed.

### Question 2 - Files created by `dvc init`

- `.dvc/config` is the project dvc configuration (remotes, default remote, core
  options). It starts empty and is **committed**, so everyone who clones the repo gets the
  same configuration.
- `.dvc/.gitignore` tells git to ignore dvc's local-only folders and files: `/cache`
  (the local copy of every data version, identified by content hash), `/tmp` (lock files,
  state database) and `/config.local` (private settings such as credentials). It is
  **committed**. What it ignores must **not** be pushed.
- `.dvcignore` works like `.gitignore`, but for dvc: files dvc should skip when it hashes
  directories. It is **committed**.

So `.dvc/config`, `.dvc/.gitignore` and `.dvcignore` go to git. `.dvc/cache`, `.dvc/tmp`
and `.dvc/config.local` stay on the machine.

### Question 3 - Where are the credentials stored? Other options than `--global`? Should they be pushed?

With `--global`, `dvc remote modify` writes to the **user-level** dvc config, outside the
repo:
- Windows: `%LOCALAPPDATA%\iterative\dvc\config`
- Linux: `~/.config/dvc/config`
- macOS: `~/Library/Application Support/dvc/config`

The password is stored in plain text there. It applies to every dvc project of that user.

The config levels, from lowest to highest priority:

| Flag | File | Committed? |
|---|---|---|
| `--system` | machine-wide config, for all users | no |
| `--global` | user config (above) | no |
| `--project` (the default when no flag is given) | `.dvc/config` | **yes** |
| `--local` | `.dvc/config.local` (ignored by `.dvc/.gitignore`) | no |

**Credentials must never be pushed to GitHub.** Anyone who can read the repo would get
access to the data, and a secret stays in git history even after it is deleted. Keep them
in `--local` (or `--global`), or in environment variables. The non-secret part (remote URL,
auth type, default remote) belongs in `.dvc/config` so a fresh clone can find the data.
That is why the remote URL was added to the project config: before, only `core.remote =
origin` was committed and the URL was only in my global config, so `dvc pull` on another
machine would not have known where `origin` was.

### Question 4 - `.gitignore` after `dvc add data`

`dvc add data` appended `/data` to the root `.gitignore`. dvc now manages the folder:
- its contents are hashed and copied into `.dvc/cache`
- git is told to ignore the real files, so 1.19 GB of images never enters git history

Git only tracks the small pointer file that describes the folder.

### Question 5 - The `.dvc` file

Yes, `data.dvc` was created. It is a small YAML pointer:

```yaml
outs:
- md5: a3a457d03c51ff8b037a833440f6ad13.dir
  size: 1188442712
  nfiles: 16643
  hash: md5
  path: data
```

`md5 ... .dir` is the hash of a manifest that lists the hash of every file in the folder,
so any change to any image gives a new hash. `size` and `nfiles` describe the folder
(1.19 GB, 16,643 images), and `path` is the tracked folder. Committing this file to git is
what versions the data: each git commit points to one exact version of the data in dvc's
cache and remote.

### Question 6 - GitHub vs DagsHub

On GitHub the code is there (`src/`, `pyproject.toml`, `uv.lock`, ...) but the images are
not. The `data` folder is ignored. What GitHub has is `data.dvc` (the pointer to the data
version) and `.dvc/config` (which says which remote the data lives in). On DagsHub, after
`dvc push`, the data shows up in the repo's file browser: the `data` folder with
`food11_raw/{training,validation,evaluation}`, marked as dvc-tracked. Git hosts the
"what"; the dvc remote hosts the "bytes".

### Question 7 - Cloning into a new folder

After `git clone` there is **no `data` folder**: only `data.dvc`, `.dvc/config` and the
code. The data is fetched with:

```bash
uv run dvc pull
```

`dvc pull` = `dvc fetch` (download the missing files from the remote into `.dvc/cache`)
+ `dvc checkout` (rebuild `data/` from the cache to match `data.dvc`). DagsHub credentials
have to be set up on that machine first, with `--local` or `--global`.

### Question 8 - Checking out an older commit

```bash
git log --oneline -- data.dvc   # 2625e5e Track data folder with dvc, then the commit that added the processed data
git checkout 2625e5e
uv run dvc checkout
```

**No.** `food11_processed` and `food11_processed_mini` disappear. Only `food11_raw` is
left, because the old `data.dvc` describes the folder as it was before `data.py` ran. dvc
restored `data/` to that version from the cache. Running `git checkout main` and then
`dvc checkout` brings both folders back. git moves the pointer, and dvc makes the files
match it.
