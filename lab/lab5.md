# Lab 5 - CI/CD with GitHub Actions

## What was done

| Step | Where |
|---|---|
| Pure, testable data functions `category_from_filename`, `resize_image` | [`src/food11/data.py`](../src/food11/data.py) |
| Module-level `model`, `SKIP_MODEL` skips the MLflow load, `/predict` answers 503 without a model | [`src/food11/serve.py`](../src/food11/serve.py) |
| Unit tests (data) and API test with a fake model, fixture image in git | [`tests/`](../tests) |
| Data contract (5 checks, exit 1 on failure) | [`src/food11/validate_data.py`](../src/food11/validate_data.py) |
| CI: lint + tests -> container smoke test -> push to ghcr.io on `main` | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| Branch protection on `main` (PR + `test` and `smoke` required) | repo settings, PR [#1](https://github.com/marcbouloss/mlops-lab-1/pull/1) |
| Image on ghcr.io, tagged with the commit SHA, pulled on the laptop | [`q7_docker_pull.txt`](../reports/lab5/q7_docker_pull.txt) |
| Promotion gate (`champion` / `challenger`) | [`src/food11/promote.py`](../src/food11/promote.py) |
| Command outputs for every question | [`reports/lab5/`](../reports/lab5) |

### Changes compared with the lab sheet, and why

1. **Food-11 contains duplicate images across splits, and `data.py` now drops them.** The very
   first run of the contract (before any planted copy) already failed check 4 on the mini
   dataset with 23 duplicates. Hashing `data/food11_raw` shows the duplicates are in the
   **original dataset**: 16 643 files but only 16 020 distinct contents, **611** of them
   present in two splits (e.g. `validation/1_129.jpg` and `training/1_328.jpg` have the same
   md5) ([`q3_validate_before_dedup.txt`](../reports/lab5/q3_validate_before_dedup.txt)).
   `data.py` now hashes every raw file and keeps only the first copy
   (training < validation < evaluation), which drops 623 files (611 cross-split + 12 within a
   split). Both datasets were rebuilt, pass all 5 checks, and the new version is in
   `data.dvc` and pushed to the DagsHub remote
   ([`q3_validate_after_dedup.txt`](../reports/lab5/q3_validate_after_dedup.txt)).

   | | training | validation | evaluation |
   |---|---|---|---|
   | `food11_processed` | 9861 | 3376 | 2783 |
   | `food11_processed_mini` | 1100 | 1096 | 1096 |

2. **`serve.py` keeps the `lifespan` handler** (Lab 3) instead of `@app.on_event("startup")`,
   which is deprecated in FastAPI. Same behaviour: the global `model` is only loaded when
   `SKIP_MODEL` is not set.
3. **The ruff rule set is pinned** in `pyproject.toml` (`select = ["E4", "E7", "E9", "F"]`, ruff's
   classic defaults). Ruff 0.17 turned on a much larger default set (bugbear, pylint, import
   sorting...) that reported 14 style findings in the Lab 1-2 code, among them B008 on
   FastAPI's standard `File(...)` default. Pinning means a ruff upgrade cannot silently change
   what CI enforces.
4. **Training on Windows**: MLflow prints `🏃 View run ...` when a run ends; with output piped,
   Windows' cp1252 console cannot encode the emoji and the script crashed *after* everything
   was logged, leaving the run in `RUNNING`. The runs were marked `FINISHED` afterwards with
   `MlflowClient().set_terminated(run_id)`; running with `PYTHONIOENCODING=utf-8` avoids it.

## Questions

### Question 1 - Why not load the real model in a unit test?

A unit test should fail only when **my code** is wrong. Loading `models:/food11@champion`
makes the result depend on things outside the code:

- **The MLflow server is not reachable.** In CI the runner is a fresh VM in GitHub's cloud;
  `http://127.0.0.1:5000` is the runner itself, not my laptop, so the load fails with a
  connection error. Same on my laptop when I forget to start the server.
- **Someone moves the alias.** `champion` is a mutable pointer: after a promotion the test
  loads another model, which may predict another class for `bread.jpg` (the test then fails
  on `category == "Bread"`) or have another output format, without a single line of code
  changing.
- **It is slow and heavy.** Downloading ~50 MB of artifacts per test run, plus torch loading
  the model, turns a 1-second test into a long one, and network hiccups or timeouts make it
  flaky.
- **Credentials / environment.** A hosted registry needs a token; a model exported on a GPU
  machine may not load on a CPU-only runner.

### Question 2 - What does `test_predict_returns_valid_json` prove?

**It proves** that everything *around* the model works: the route `/predict` exists and
accepts a multipart upload; a real JPEG is decoded by PIL; `EVAL_TF` turns it into a
`(1, 3, 224, 224)` array that is passed to `model.predict`; the output is turned into
probabilities (softmax), the arg-max index is mapped to the right name in `CLASSES`, and the
answer is valid JSON with a `category` and a `confidence` in [0, 1] and status 200.

**It does not prove** anything about the model: not that it is accurate (the "Bread" answer
is hard-coded in the fake), not that the real model accepts this input shape/dtype or returns
this output shape, not that it can be loaded from MLflow, and not that the preprocessing
matches what was used in training. The confidence check is weak too: the fake returns
`[0.90, 0.01, ...]` and `serve.py` applies a softmax, so the API answers about **0.196**, not
0.90, and the test still passes.

### Question 3 - Which check caught the planted duplicate? Why is it worse than a corrupted image?

Check 4, **"no image appears in two splits"**: it compares the md5 of the file content, so
the copy is found even under another name
([`q3_validate_planted_duplicate.txt`](../reports/lab5/q3_validate_planted_duplicate.txt)):

```
FAIL  no image appears in two splits
      - data\food11_processed_mini\evaluation\Bread\planted_copy.jpg duplicates data\food11_processed_mini\training\Bread\0_103.jpg
exit status: 1
```

A corrupted image is **loud**: training crashes, you see it at once and fix it. A
train/test duplicate is **silent**: everything runs, but the model is graded on images it has
already memorised, so `test_accuracy` is too optimistic. That wrong number is then used to
decide what goes to production (the promotion gate below compares exactly this metric), and
you only discover the truth when real users send images the model has never seen.

This is not hypothetical here: before de-duplication, 23 images of the mini evaluation /
validation splits were copies of images from another split (611 in the full dataset), so the
Lab 2 `test_accuracy` of the `champion` (0.827) was measured on a partly leaked test set.

### Question 4 - Where should `validate_data.py` run if CI cannot see the data?

1. **In the data pipeline, where the data is**: as a DVC stage between building the
   dataset and training (`deps: data/food11_processed_mini`, `cmd: python src/food11/validate_data.py ...`).
   `dvc repro` then stops before training on data that breaks the contract, on whatever
   machine trains. The same command can run as a pre-commit / pre-push hook before
   `dvc add` + `dvc push`, so broken data never reaches the remote.
2. **In CI, after pulling the data from the remote**: the DVC remote is on DagsHub, which a
   GitHub runner can reach. A job (or the training job) with a DagsHub token in the repo
   secrets runs `dvc pull data` (or only the mini dataset) and then `validate_data.py`, on
   every change of `data.dvc` or on a schedule.

### Question 5 - `needs: test`

If a unit test fails, the `test` job fails and **`smoke` is skipped** (it never starts);
`push`, which needs `smoke`, is skipped too, so no image is built or published. Without
`needs`, jobs are independent and start **in parallel** on separate runners: `smoke` would
build the image even though the tests are red (wasted minutes), and with `push` depending only
on `smoke`, an image whose tests fail could still be pushed to the registry.
The broken-test pull request below shows it: `test` failed and `smoke` was skipped.

### Question 6 - Duration of the run, first vs second

([`q6_ci_timings.txt`](../reports/lab5/q6_ci_timings.txt), runs on `main`, all three jobs)

| | run 1 (`9ae1ca8`, first run) | run 2 (`c71f827`, empty commit) |
|---|---|---|
| whole run (wall clock) | **610 s** (~10 min) | **177 s** (~3 min) |
| `test` job (`uv sync` / `pytest`) | 25 s (5 s / 12 s) | 24 s (5 s / 12 s) |
| `smoke` job (`docker build`) | 133 s (118 s) | 122 s (105 s) |
| `push` job (`docker/build-push-action`) | 445 s (**416 s**) | 26 s (**5 s**) |

The step that got faster is **`docker/build-push-action`: 416 s -> 5 s**. Run 1 built every
layer (the `uv sync` layer downloads torch) and exported them to the GitHub Actions cache
(`cache-to: type=gha,mode=max`). Run 2 builds the same `pyproject.toml`/`uv.lock` and `src/`,
so with `cache-from: type=gha` every layer is a cache hit, and the registry already has
those layers, so nothing is uploaded either. The plain `docker build` of the `smoke` job does
**not** get faster (118 s -> 105 s): it has no cache configured, and every runner starts
empty. `uv sync` was already fast in run 1 (5 s): uv downloads in parallel, and the CPU
build of torch is used on Linux, not the multi-GB CUDA one (`[tool.uv.sources]`).

### Question 7 - Why tag with the commit SHA rather than `latest`?

The SHA is **immutable and traceable**: `ghcr.io/marcbouloss/mlops-lab-1:<sha>` always means
the image built from that exact commit, so you know which code is running, can reproduce a
bug with the same image, and roll back by deploying the previous SHA. `latest` is a
**mutable** pointer that moves on every push to `main`, and `docker run` does not re-pull an
image it already has.

Concrete case: on Monday the laptop and the server both `docker pull ...:latest` (commit A).
On Tuesday a commit B that changes the preprocessing is merged and CI pushes it as `latest`.
The server is restarted and pulls again, so it now runs **B**; the laptop still has the
cached `latest` and runs **A**. Both say "latest", they give different predictions for the
same image, and nobody can tell from the tag which code is behind each. With SHA tags the
compose file names one exact image, and both machines run the same code by construction.

The image of the first run is on ghcr.io and was pulled on the laptop
([`q7_docker_pull.txt`](../reports/lab5/q7_docker_pull.txt)):

```
$ docker pull ghcr.io/marcbouloss/mlops-lab-1:9ae1ca8c6e390eb24e834bf4e7d5ef5abdaa90ea
Digest: sha256:ad322ba23a9f72d604159aff8cc14680c4769f2cad82aa6c6b7234942db9e779
ghcr.io/marcbouloss/mlops-lab-1:9ae1ca8c6e390eb24e834bf4e7d5ef5abdaa90ea   ad322ba23a9f   2.57GB
```

### Question 8 - Why does `push` declare `permissions:` explicitly?

Least privilege. `GITHUB_TOKEN` can carry write access to the whole repository; the `push`
job only needs to **read** the code and **write packages**, so that is all it gets
(`contents: read`, `packages: write`); every other scope is `none`. Every step of the job runs
with that token in reach, including third-party actions (`docker/*`) and everything executed
during `docker build`, where any compromised dependency could run code.

With a token that could write everything, a compromised step could push a commit to `main`
(a backdoor in `serve.py` that the next deploy ships), move or create tags and releases,
delete branches, change or merge pull requests, open/close issues, or overwrite published
images. With the declared permissions the worst case is limited to the package.

### Question 9 - The pull request with a broken test

Rule on `main`: pull request required, status checks `test` and `smoke` required,
**"include administrators"** on (otherwise I, as repo owner, could bypass it). Required
approvals are set to 0 because GitHub does not let you approve your own PR on a one-person
repo.

On branch `lab5-broken-test`, `category_from_filename` shifts the index by one. PR
[#1](https://github.com/marcbouloss/mlops-lab-1/pull/1) then shows
([`q9_pr_broken.txt`](../reports/lab5/q9_pr_broken.txt)):

```
test   fail      (twice: once for the push event, once for the pull_request event)
smoke  skipping  (needs: test)
push   skipping
mergeStateStatus: BLOCKED
E       AssertionError: assert 'Dairy product' == 'Bread'
FAILED tests/test_data.py::test_category_from_filename
$ gh pr merge 1 --merge
X Pull request marcbouloss/mlops-lab-1#1 is not mergeable: the base branch policy prohibits the merge.
```

The page shows the red check with a link to the failing log, "Merging is blocked", and a
greyed-out merge button: **I could not merge**. After pushing the fix both checks turned
green, the state became `CLEAN` and the PR was merged
([`q9_pr_fixed.txt`](../reports/lab5/q9_pr_fixed.txt)).

For a team, "everybody runs the tests on their laptop" is a **convention**: it relies on
people remembering, on their local environment (another Python, a package installed by
hand, data on their disk) and on trust. The protected branch makes it a **rule** that is
checked by a machine in the same clean environment for every change, and the result is
visible on the PR for the reviewer. Nothing reaches `main`, and so no image is pushed, unless
lint, tests and the container smoke test pass.

### Question 10 - Gate decisions

Two new runs on the de-duplicated mini dataset (resnet18, 5 epochs, batch 32), registered as
`food11` v2 and v3. The champion is v1 (Lab 2/3 run `cc8c24cb`, lr 1e-4). `n` = 1096 images in
`food11_processed_mini/evaluation` ([`q10_promote.txt`](../reports/lab5/q10_promote.txt)):

```
se        = sqrt(0.8273 * (1 - 0.8273) / 1096) = 0.0114
threshold = 0.8273 + 2 * 0.0114             = 0.8501
```

| candidate | lr | `test_accuracy` | vs threshold 0.8501 | decision |
|---|---|---|---|---|
| v2 | 1e-6 (very small) | 0.1359 | far below | **challenger**, `champion` stays on v1 |
| v3 | 1e-4 (best Lab 2 settings) | 0.8358 | below (+0.85 points over the champion) | **challenger**, `champion` stays on v1 |

After both runs: aliases `champion -> 1`, `challenger -> 3`. Each candidate carries the tags
`gate_decision`, `gate_acc_candidate`, `gate_acc_champion`, `gate_champion_version`,
`gate_threshold`, `gate_n_test`.

With n = 1096, **2 * se = 0.0228, about 2.3 accuracy points**. A new model must win by more
than that before it replaces the champion. v2 is clearly worse: with lr = 1e-6 the weights
barely move in 5 epochs (val_accuracy 0.09 -> 0.14). v3 is better on paper, but 0.85 points
is well inside the noise of a 1096-image test set, so the gate keeps the current model and
v3 waits as `challenger`. That is the point of the gate: it stops a coin-flip difference from
changing what is in production.

One caveat: v1's 0.8273 was measured in Lab 2 on the mini evaluation split **before**
de-duplication, which was slightly leaked (Question 3), while v2/v3 are measured on the clean
split. A fair comparison re-evaluates the champion on the same test set as the candidate
before applying the gate.

### Question 11 - Running `promote.py` automatically

`promote.py` needs the registry, and a GitHub-hosted runner cannot reach `127.0.0.1:5000` on
my laptop. Two ways:

1. **Move MLflow to a hosted server: DagsHub.** The repository already has a DagsHub mirror
   (it is the DVC remote), and DagsHub provides an MLflow server per repo
   (`https://dagshub.com/<user>/mlops-lab-1.mlflow`). With `MLFLOW_TRACKING_URI`,
   `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` stored as GitHub secrets, a
   workflow job (`workflow_dispatch` with the version as input, or triggered after a training
   job) runs `promote.py` on a normal runner.
2. **Run it where MLflow and the GPU are**: register a **self-hosted GitHub runner** on the
   training machine, so a workflow job can reach `localhost:5000` (train -> register ->
   `promote.py` in one job). Or, without GitHub, make it the last step of training: a DVC
   stage after `train`, or a call at the end of `train.py` after `register_model`.

### Question 12 - The pipeline from `git push` to a running container

`[A]` = automatic, `[H]` = needs a human.

```
 CODE VERSION (git commit SHA)                         MODEL VERSION (registry alias)
 ============================                          ==============================
 [H] edit code, git push (branch)                      [H] dvc repro / train.py on the GPU laptop
            |                                                     |  data contract: validate_data.py [H, local]
            v                                                     v
 GitHub Actions  ci.yml                                [A] MLflow run: params, metrics, model
 +------------------------------------+                           |
 | [A] test : uv sync, ruff, pytest   |                           v
 | [A] smoke: docker build,           |                [H] register as food11 vN
 |            SKIP_MODEL=1 + /health  |                           |
 +------------------------------------+                           v
            |  green checks on the PR                  [H] run promote.py --version N
            v                                              [A] gate: acc >= champion + 2se ?
 [H] merge the PR into main  (branch protection:              yes -> alias champion = vN
     PR + test + smoke required)                              no  -> alias challenger = vN
            |                                                    + gate_* tags on vN
            v                                                     |
 [A] push job (main only):                                        |
     docker build-push -> ghcr.io/marcbouloss/mlops-lab-1:<sha>   |
            |                                                     |
            |   <-- the code version is decided HERE              |   <-- the model version is
            |       (the SHA that was merged to main)             |       decided HERE (alias)
            v                                                     v
 [H] docker pull ...:<sha> ; docker compose up      --->  container starts, serve.py loads
                                                          models:/food11@champion from MLflow
                                                                  |
                                                                  v
                                                  running container = image <sha> + model @champion
```

- **Automatic**: lint, unit tests, container smoke test, the merge block on red checks,
  building and pushing the SHA-tagged image, logging runs to MLflow, the gate's decision
  and its tags.
- **Still human**: writing and merging the code, training (GPU laptop, not in CI), running
  `validate_data.py`, registering the model, launching `promote.py`, and deploying
  (`docker pull <sha>` + `docker compose up`, or restarting the inference container so it
  reloads `@champion`).
- The **code version** is decided by the merge to `main`: the image is tagged with that
  commit's SHA. The **model version** is decided independently in the registry, by the
  `champion` alias that `promote.py` moves. The same image can serve a new model after a
  restart, and a new image can serve the same model.
