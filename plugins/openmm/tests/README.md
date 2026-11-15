# The plugin's tests

Correctness tests of the plugin against stock OpenMM, and its benchmarks. Run
them from the root of the source tree, which enters the plugin's container by
itself (`./run_docker`, built from the `Dockerfile` there):

```sh
./run_test openmm_unit                   # build the plugin, then its tests
./run_test openmm_unit -k triclinic      # the same, with a pytest filter
./run_test openmm_sample                 # the solvated_systems sample
./run_test openmm_bench                  # the benchmarks
./run_test openmm_bench -k "dhfr and nvt and mixed"   # one system, ensemble, precision
./run_test --list-stages                 # every stage there is
```

The whole `openmm_unit` stage takes a few hours on one GPU, `openmm_sample` about
a minute, and `openmm_bench` about an hour. The docs site's
[Tests](https://docs.nvidia.com/cuda/cupotential/tests) page covers correctness
checks; the [Overview](https://docs.nvidia.com/cuda/cupotential#benchmarks)
contains benchmark results and instructions.

`NV_GPU=0 ./run_test ...` selects the first host GPU. GPU indices start at 0.
If `NV_GPU` is unset, all GPUs are available to the Docker container. The tests
fail at once if another process is using the GPU they run on.

> [!TIP]
> If the GPU disappears in the middle of the tests, with "Failed to initialize NVML:
> Unknown Error" or "No compatible CUDA device is available", the host probably ran
> `systemctl daemon-reload`. Set `CUPO_DOCKER_ADD_DEVICE_NODES=1` to have
> `./run_docker` also pass the GPUs' device nodes, which keeps them. See
> [The GPUs the container gets](https://docs.nvidia.com/cuda/cupotential/install#the-gpus-the-container-gets).

The stage list is the one this source tree was cut from, which has more than
this tree: a stage whose files are not here is listed with `#` and skipped,
saying "not available in this tree (no <file>)". `cupo_build` is one of them:
cupo is not built here, and `openmm_build` builds the plugin against cupo's
libraries in `build/` (from the release, see `plugins/openmm/run_build`).

## What is skipped here, and why

- **Benchmark comparisons.** The benchmarks compare each measurement with
  committed reference numbers for the GPU it ran on, and this tree carries none.
  Each benchmark runs, records what it measured under
  `plugins/openmm/build/bench-results/`, and is skipped with "measured, not
  compared".
- **Datasets.** The tests over the solvated systems in `data/` run on every
  dataset that is there (`data/PROVENANCE.md`); a dataset that is not there is
  not tested. apoa1 and dhfr are OpenMM's own benchmark systems and are not in
  `data/`: the tests read them from your OpenMM installation
  (`share/openmm/examples/benchmarks/`, which conda-forge's `openmm` installs),
  and when it has none, download them once from OpenMM's GitHub repository into
  `.tmp/openmm-benchmarks/`, checked by SHA-256
  (`plugins/openmm/python/nv_openmm_plugin/_benchmark_systems.py`).
  With `CI` set they are not downloaded, and a missing file fails the test.
  `CUPO_OPENMM_BENCHMARKS_DIR` names another directory to read them from.

## The GPU clocks

The benchmarks lock the GPU's clocks for the session (`tests/gpu_clocks.py`, at
the values in `tests/hw_config.py`). That takes passwordless `sudo nvidia-smi`,
which the container has, and the `SYS_ADMIN` capability, which `./run_docker`
gives the container. When the clocks cannot be locked, the benchmarks fail,
saying so and how to fix it.

`CUPO_ALLOW_UNLOCKED_CLOCKS=1 ./run_test openmm_bench` runs them at whatever clocks
the GPU has instead. They measure and print, marked "clocks not locked", and are
skipped with "measured, not compared": an unlocked number is compared with
nothing, even where reference numbers exist.

## Testing the installed plugin

By default the tests test the plugin the tree built: `plugins/openmm/python/` and
`plugins/openmm/build/`. With `CUPO_TEST_INSTALLED=1` they test the plugin pip or
conda installed instead, which they import from wherever it was installed, and
`openmm_build` is skipped. Run them in the Docker environment, in the
`./run_docker bash` session that installed it.

After a pip install, the tests run with the Docker environment's Python, which
has pytest:

```bash
./run_docker bash
pip install <the release's wheel>
CUPO_TEST_INSTALLED=1 ./run_test openmm_unit -k test_unit_conversion
```

After a conda install, the plugin is in the micromamba environment
`cupotential`, so the tests run in that environment, with `micromamba run`. The
environment needs pytest, which the install does not bring:

```bash
./run_docker bash
micromamba create -y -n cupotential -c ./<the release's conda channel> -c conda-forge nv-openmm-plugin
micromamba install -y -n cupotential -c conda-forge pytest
CUPO_TEST_INSTALLED=1 micromamba run -n cupotential ./run_test openmm_unit -k test_unit_conversion
```

`-k test_unit_conversion` runs one short test module, in under a minute; without
`-k`, the whole stage takes a few hours. Outside Docker, the environment the
plugin is installed in needs pytest too.

The session's header says which plugin it tests, "installed" or "tree build",
with its module, its library directory and its version. With the variable set
and no installed plugin, or with the tree's build in its place, the session fails
at once, saying so. A test that needs the tree's own build skips, naming the
variable.
