# cua-speedrun

Compare computer-use agents by performance, time, and cost on real desktop tasks.

## Quickstart

Install and configure your credentials:

```bash
pip install cua-speedrun
cua-speedrun setup
cua-speedrun benchmark
```

Choose an agent template and benchmark, review or edit the configuration,
then start. The terminal shows preparation progress and opens the evaluation
dashboard. `setup` lets you add Modal credentials, API keys, and other variables.

Prebuilt desktop images are imported from Docker Hub and cached in Modal.
Models and application setup are cached on first use. Data lives in
`~/.local/share/cua-speedrun`; set `CUA_SPEEDRUN_HOME` to use another location.
Use `cua-speedrun doctor` to inspect your installation.

## Run from a Modal notebook

Open [`notebooks/run-on-modal.ipynb`](notebooks/run-on-modal.ipynb) in a
[Modal notebook](https://modal.com/docs/guide/notebooks) and run its one cell:

```python
%uv pip install -q -U "cua-speedrun>=0.3.13" ipywidgets
from cua_speedrun.notebook import dashboard
dashboard()
```

Choose an agent or upload your own `agent.py`, pick a task set, add your keys,
and start. Evaluations run in the Modal account that runs the notebook, the
same as `--host modal` below, and keep going after the notebook stops.

## Run an evaluation

Evaluations run on Modal by default, using credentials from `setup`, your
Modal CLI profile, or environment variables. For an API agent:

```bash
export ANTHROPIC_API_KEY="your-api-key"
cua-speedrun benchmark --dataset osworld-50 --agent claude
```

For Claude Code, use `--agent claude_code` with `ANTHROPIC_API_KEY` or
`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. If both are set, the OAuth
token is used. Set `CLAUDE_CODE_MODEL` and `CLAUDE_CODE_EFFORT`, and pass them
with `--env CLAUDE_CODE_MODEL --env CLAUDE_CODE_EFFORT` to select the model and effort.

For Yutori n2, use `--agent yutori_n2` with `YUTORI_API_KEY`.
Set `YUTORI_REASONING_EFFORT` to `none`, `low`, `medium`, or `xhigh` to choose
the reasoning effort; the default is `medium`.

Supplying both `--agent` and `--dataset` starts the evaluation directly.
Run `cua-speedrun benchmark` without them for guided configuration.
For scripts and AI agents, `--json` writes JSON records
to stdout; preparation logs go to stderr:

```bash
cua-speedrun --json benchmark --dataset osworld-50 --agent qwen3vl
```

The command follows progress until the evaluation finishes; add `--background`
to return the evaluation ID once launched. `--json` also
works with `setup`, `catalog`, `status`, and `evaluations`.
To launch and inspect evaluations in a browser, run `cua-speedrun dashboard`.

To keep an evaluation running when your computer disconnects, choose **Modal**
under **Run controller on**, or add `--host modal`:

```bash
cua-speedrun benchmark --host modal --dataset osworld-50 --agent claude
cua-speedrun evaluations --host modal
cua-speedrun status RUN_ID
cua-speedrun cancel RUN_ID
cua-speedrun export RUN_ID
```

Each hosted evaluation uses a CPU controller in your Modal account and saves
logs and results to a Modal Volume. The controller stops when the evaluation
finishes. `status` reconnects to the terminal dashboard; `export` downloads the
saved trajectories. Closing the dashboard detaches without cancelling the run.

- Add `--parallel-evaluations 4` to run four agent replicas in parallel.
- Bundled GPU agents select their GPU automatically; override it with `--gpu L40S`.
- Add `--no-preload` to disable environment preloading.
- To use local Linux hardware, add `--compute local --environment local`.
  Local desktops require KVM/QEMU.

Inspect results or download trajectories:

```bash
cua-speedrun evaluations
cua-speedrun status RUN_ID
cua-speedrun export RUN_ID
```

Use `cua-speedrun catalog` to list available agents and benchmarks, or
`cua-speedrun help benchmark` for more options.

## Benchmarks

| Benchmark | Tasks |
| --- | ---: |
| `cua-world-26` | 26 |
| `osworld-50` | 50 |
| `osworld2-52` | 52 |
| `my-pc-bench` | 38 |
| `cua-world-offline` | 143 |
| `osworld-offline` | 295 |
| `osworld2-offline` | 63 |

The `offline` variants contain the full offline task sets; the smaller variants
are representative subsets. MyPCBench also requires a
`MYPCBENCH_JUDGE_API_KEY` for its evaluator; see its
[setup instructions](benchmarks/my-pc-bench/README.md).

## Bring your own agent

Start from an implementation in [`agents/`](agents/). Each agent has two files:

- `init.py` prepares dependencies or starts a model server before task timing begins.
- `agent.py` receives the environment URL and task description, then interacts
  through [`Computer`](src/cua_speedrun/client.py).

Submit the folder directly:

```bash
cua-speedrun validate --agent ./my-agent
cua-speedrun benchmark --dataset osworld-50 --agent ./my-agent
```

Additional packages can be installed by `init.py`; the submission uploads
`init.py` and `agent.py`. An optional `agent.json` declares `gpu` and
`required_environment_variables`. Use `optional_environment_variables` for
variables forwarded when set. To contribute an agent, add its folder to
`agents/`; it is discovered automatically.

## Bring your own benchmark

A benchmark is a folder with a `manifest.yaml`, task folders containing
`task.yaml`, and its environment setup and verifier. The manifest lists tasks:

```yaml
name: my-benchmark
version: "1"
tasks: [tasks/my-task]
```

Each `task.yaml` specifies `task_id`, `description`, and an `env` mapping:

```yaml
task_id: my-task
description: The task for the agent to complete.
env:
  kind: gym-anything
  env_dir: ${BENCHMARK_DIR}/environment
  task_id: my-task
```

Keep the [Gym-Anything environment](https://github.com/cmu-l3/gym-anything)
and its task setup/verifier inside the benchmark folder. Then:

```bash
cua-speedrun validate --dataset ./my-benchmark
cua-speedrun benchmark --dataset ./my-benchmark --agent ./my-agent
```

The benchmark is copied into the installation. Increase its version when
changing a registered task set. To contribute it, add the folder under
`benchmarks/` and its name to `catalog/benchmarks.yaml`; packaging is automatic.

## Repository structure

- [`agents/`](agents/) — agent implementations.
- [`benchmarks/`](benchmarks/) — task sets and benchmark definitions.
- [`src/cua_speedrun/`](src/cua_speedrun/) — execution, timing, scoring, and the dashboard.
- [`src/cua_speedrun/compute_runners/`](src/cua_speedrun/compute_runners/) — local and Slurm runners.
