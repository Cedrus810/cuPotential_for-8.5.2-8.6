# nv-openmm-plugin 0.3.5+openmm86：OpenMM 8.6 版（实验室内部分发）

这是在 NVIDIA 官方 cuPotential **v0.3.5** 的基础上，为 **OpenMM 8.6** 重新编译、并做了一处修正的版本，只在实验室内部分发。它**不是 NVIDIA 官方发布**：官方 0.3.5 只支持 OpenMM 8.4，PyPI 上的 `nv-openmm-plugin` 也是官方版。

> 状态（2026-11-24）：方式 A（全新环境 + pip）已经在干净环境里实际装过、测过，见 §5。方式 B 还没测，标了【待验证】。

## 1. 依赖条件

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Linux x86_64 | wheel 的平台标签是 `linux_x86_64`，不是 manylinux |
| Python | **3.14**（只能 3.14） | 插件带有预编译的 SWIG 扩展，wheel 标签是 `cp314-cp314`，换别的 Python 版本装不上 |
| OpenMM | **8.6.x**（`>=8.6,<8.7`），要带 CUDA 13 平台 | wheel 写明依赖 `openmm[cuda13]>=8.6,<8.7`。插件在 import 时还会再检查一遍：运行中的 OpenMM 不在 8.6 系列就拒绝加载 |
| CUDA 运行库 | CUDA **13.4.x**：`nvidia-cuda-runtime`、`nvidia-cuda-nvrtc`、`nvidia-nvjitlink` 都要 `>=13.4,<13.5`，`nvidia-cufft` 要 `>=12.4,<12.5` | 用 pip 安装时会自动装上。cupo 的嵌入式 LTO IR 和 NVRTC 的版本是绑在一起的，CUDA 小版本不能换 |
| NVIDIA 驱动 | 要支持 CUDA 13.4 | OpenMM 用 NVRTC 现场编译内核，驱动比 NVRTC 旧时会报 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`。已验证的是 615.71；580.x 没测过 |
| GPU 架构 | sm_75 / 80 / 86 / 89 / 90 / 100 / 103 / 120 | 这是 cupo 运行时编译时包含的架构，即 Turing 到 Blackwell。实测只在 RTX 5090（sm_120）上跑过 |

**不要**和官方的 `nv-openmm-plugin`（0.3.5，对应 OpenMM 8.4）装在同一个环境里，两者包名相同。如果已经装了官方版，先执行 `pip uninstall nv-openmm-plugin`。

## 2. 安装

### 方式 A：全新环境，用 pip（已验证，推荐）

```bash
# 准备一个 Python 3.14 的环境，用 venv 或 conda 都可以
micromamba create -y -n omm86 -c conda-forge python=3.14 pip
micromamba activate omm86
pip install ./nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
```

pip 会按 wheel 里写的依赖，从 PyPI 装 `openmm[cuda13]` 8.6.x 和 CUDA 13.4 的运行库。2026-11-24 实测装上的版本：`openmm 8.6.1`、`OpenMM-CUDA-13 8.6.1`、`nvidia-cuda-runtime / nvrtc / nvjitlink 13.4.92`、`nvidia-cufft 12.4.0.43`。`OpenMM-CUDA-13` 还会带进 `nvidia-cuda-nvcc`、`nvvm`、`crt`、`cupti`，版本是 13.5.x，这不影响使用。在新 shell 里，PATH 上不用有 CUDA toolkit，也不用设 `LD_LIBRARY_PATH`：运行时加载的 libcudart、libcufft、libnvrtc、libnvJitLink 都是 pip 装在 `site-packages/nvidia/cu13/lib/` 下的那份。

如果没有 micromamba 或 conda，用任意一个 Python 3.14 建 venv 也可以（实测就是这么做的）：

```bash
python3.14 -m venv omm86 && source omm86/bin/activate
pip install --upgrade pip
pip install ./nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
```

### 方式 B：已有 conda-forge 的 OpenMM 8.6 环境【待验证】

如果环境里的 OpenMM 8.6.x 是从 conda-forge 装的（`openmm=8.6 cuda-version=13.4`），不想让 pip 再从 PyPI 装一份 OpenMM，可以这样：

```bash
pip install --no-deps ./nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
```

这种方式要求环境里已经有 CUDA 13.4 的 cudart、nvrtc、nvjitlink，以及 cuFFT 12.4（比如 conda-forge 的 `cuda-cudart`、`cuda-nvrtc`、`libnvjitlink`、`libcufft`）。

## 3. 检查是否装好

```bash
python -c "import openmm, nv_openmm_plugin as nv; print(openmm.version.full_version); print([f for f in openmm.Platform.getPluginLoadFailures() if 'nv_openmm_plugin' in f])"
```

应该打印出 8.6.x 的版本号，以及一个空列表。

## 4. 和官方 0.3.5 的区别

1. **适配 OpenMM 8.6**：插件源码不用改，重新编译即可。构建时会自动探测 OpenMM 的系列，依赖钉扎和加载时的版本检查都按 8.6 生成。
2. **修正 OpenMM 8.5 以后的能量最小化**：OpenMM 8.5 起，能量最小化改在 GPU 上做。对于有约束的体系，约束弹簧的力会越来越大，最终溢出 GPU 的定点数力缓冲。溢出后 OpenMM 会在 CPU 上用整个 System 新建一个 Context 来算，而插件的力在 CPU 上没有实现，于是报错 `Specified a Platform for a Context which does not support all required kernels`。本版本的处理是：
   - System 里**没有**插件的力：照常用 OpenMM 自己的 GPU 能量最小化，行为不变；
   - System 里**有**插件的力（`nv.NonbondedForce`、`nv.MonteCarloBarostat`）：按 OpenMM 8.4 的做法，在主机上跑 L-BFGS，约束弹簧在主机上用双精度计算。这是 8.4 的算法，用不上 8.5 以后的 GPU 能量最小化，所以会比它慢；
   - 体系本身的力超过约 2e9 kJ/mol/nm（比如原子重叠）时，会直接报错并说明原因，不再转去 CPU。遇到这种情况，先把结构里的冲突处理掉再做能量最小化。
3. 用法和 API 与官方 0.3.5 完全一样，可以参考官方 README 和 `samples/`。

## 5. 已验证的环境（2026-11-23～24）

| 项 | 值 |
|---|---|
| GPU / 驱动 | RTX 5090 / 615.71 |
| OpenMM | 8.6.1（conda-forge，`8.6.1.dev-b399af4`） |
| CUDA | 13.4.92 |
| Python | 3.14.7 |

在这个环境下的结果：

| 项目 | 结果 |
|---|---|
| 插件测试（非 benchmark） | 除 `test_leak_detection` 外全部通过；它需要 cupy，环境里没装 |
| `samples/solvated_systems` | 通过 |
| `samples/alchemical` | 通过 |
| benchmark（频率未锁） | 插件相对同一张卡上的原生 OpenMM 8.6.1，每步加速 2.3～3.8 倍 |

上面这些都是从**源码树**跑的。下面是从**装好的 wheel** 跑的（2026-11-24，方式 A，新建 venv，没有设 `LD_LIBRARY_PATH` 和 `PYTHONPATH`）：

| 项目 | 结果 |
|---|---|
| `import nv_openmm_plugin` | 成功，插件加载失败 0 条 |
| `samples/solvated_systems` | 通过 |
| `CUPO_TEST_INSTALLED=1` `test_unit_conversion` | 14 passed |
| `CUPO_TEST_INSTALLED=1` `test_npt` | 14 passed（覆盖能量最小化的修正） |

## 6. 构建方法（给维护者）

在 fulda 上构建，不用 docker：

```bash
E=<OpenMM 8.6 + swig + cmake 的 conda 环境>
export PATH=/usr/local/cuda-13.4/bin:$E/bin:$PATH OPENMM=$E
plugins/openmm/run_build
GITHUB_RELEASE_TAG=v0.3.5 python -m pip wheel --no-build-isolation --no-deps -w dist plugins/openmm
```

`plugins/openmm/VERSION` 写的是 `0.3.5+openmm86`。根目录的 `VERSION` 和 `RELEASE_TAG` 保持 0.3.5 / v0.3.5 不改，因为 `fetch_cupo` 要用它们去官方 v0.3.5 的 release 下载 cupo 运行时。
