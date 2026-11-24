[English](README.md) | [日本語](README.ja.md) | **中文**

# cuPotential OpenMM 插件 — OpenMM 8.6 实验室版（`0.3.5+openmm86`）

本仓库是 [NVIDIA cuPotential](https://github.com/NVIDIA/cuPotential) **v0.3.5** 的实验室 fork：OpenMM 插件针对 **OpenMM 8.6** 重新编译，并修正了一处能量最小化的问题。它**不是 NVIDIA 官方发布**。NVIDIA 的 0.3.5（包括 PyPI 上的 `nv-openmm-plugin`）对应的是 OpenMM 8.4。NVIDIA 原来的 README 保留在 [README.NVIDIA.md](README.NVIDIA.md)。API、原理和示例都没有改动，请看 [cuPotential 官方文档](https://docs.nvidia.com/cuda/cupotential)。

## 和 NVIDIA 0.3.5 的区别

- **支持 OpenMM 8.6**
  - 插件源码不用改，直接对 OpenMM 8.6 编译。
  - 编译时 CMake 会探测 OpenMM 的版本，并据此生成三样东西：
    - wheel 的依赖：`openmm[cuda13]>=8.6,<8.7`；
    - conda recipe 里的版本钉扎；
    - import 时的版本检查：运行中的 OpenMM 不是 8.6 系列就拒绝加载。
- **修正 OpenMM 8.5 以后的能量最小化**
  - 从 8.5 起，OpenMM 在 GPU 上做能量最小化。体系里有约束时，模拟约束的弹簧力最终会溢出 GPU 的定点数力缓冲。这时 OpenMM 会在 CPU 上用整个 System 重建一份，而插件的力在 CPU 上没有实现，于是报错 `Specified a Platform for a Context which does not support all required kernels`。
  - 本版本对含插件力（`nv.NonbondedForce`、`nv.MonteCarloBarostat`）的 System，改用 OpenMM 8.4 的做法：在主机上跑 L-BFGS，弹簧力也在主机上用双精度加。这是 8.4 的算法，用不上 8.5 以后的 GPU 能量最小化，所以会比它慢。
  - 不含插件力的 System 照常用 OpenMM 自己的 GPU 能量最小化。
  - 如果体系本身的力分量达到约 2e9 kJ/mol/nm（比如原子重叠），会直接报出说明原因的错误，不再转去 CPU。
- **版本号 `0.3.5+openmm86`**：加了 PEP 440 的本地标记，不会和 NVIDIA 的 0.3.5 混淆。

API 和用法与 NVIDIA 0.3.5 完全相同。

## 依赖条件

| | |
|---|---|
| 系统 | Linux x86_64 |
| Python | **只能用 3.14**（wheel 里带了预编译的 SWIG 扩展，标签是 `cp314`） |
| OpenMM | 8.6.x，要带 CUDA 13 平台（pip 会从 PyPI 自动安装） |
| CUDA 库 | CUDA runtime、NVRTC、nvJitLink `>=13.4,<13.5`，cuFFT `>=12.4,<12.5`（pip 会自动安装，不需要 CUDA toolkit） |
| NVIDIA 驱动 | 要支持 CUDA 13.4（615.71 验证过，580.x 没测） |
| GPU | sm_75～sm_120（Turing～Blackwell），实测只在 RTX 5090 上跑过 |

不要和 NVIDIA 的 `nv-openmm-plugin` 装在同一个环境里，两者包名相同。

## 安装

从本仓库的 **Releases**（tag `v0.3.5+openmm86`）下载 `nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl`，然后：

```bash
python3.14 -m venv omm86 && source omm86/bin/activate     # 也可以用 python=3.14 的 conda 环境
pip install --upgrade pip
pip install nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
python -c "import openmm, nv_openmm_plugin; print(openmm.version.full_version, [f for f in openmm.Platform.getPluginLoadFailures() if 'nv_openmm_plugin' in f])"
```

最后一条命令打印出 8.6.x 的版本号和 `[]`，就说明装好了。不需要设置 `LD_LIBRARY_PATH`。

用 `pip install --no-deps` 装进已有的 conda-forge OpenMM 8.6 环境，这种方式还没有测过。

## 已验证的环境

验证环境：RTX 5090、驱动 615.71、OpenMM 8.6.1、CUDA 13.4.92、Python 3.14.7。

| 测了什么 | 结果 |
|---|---|
| wheel 装进新建的 venv | 能 import，CUDA 库全部从 venv 里加载 |
| 用 wheel 跑 `samples/solvated_systems` | 通过 |
| 用 wheel 跑已安装插件测试 `test_unit_conversion` / `test_npt` | 通过 |
| 在源码树里跑插件测试套件 | 全部通过（需要 `cupy` 的 `test_leak_detection` 没有跑） |
| 在源码树里跑 benchmark（GPU 频率没锁） | 相对同一张卡上的原生 OpenMM 8.6.1，每步快 2.3～3.8 倍 |

Release notes（英文）：[plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md](plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md)

## 从源码构建（维护者）

需要准备：

- 一个装了 `openmm=8.6`、`python=3.14`、`swig`、`cmake>=3.24`、`cuda-version=13.4` 的 conda 环境；
- CUDA 13.4 toolkit（`nvcc`）；
- gcc。

不需要 Docker。

```bash
E=<上面那个 conda 环境>
export PATH=<cuda-13.4>/bin:$E/bin:$PATH OPENMM=$E
CUPO_REPOSITORY=NVIDIA/cuPotential python3 fetch_cupo      # NVIDIA v0.3.5 的 cupo 运行时（CUDA 13.4）
plugins/openmm/run_build
GITHUB_RELEASE_TAG=v0.3.5 python -m pip wheel --no-build-isolation --no-deps -w dist plugins/openmm
```

根目录的 `VERSION`／`RELEASE_TAG` 保持 `0.3.5`／`v0.3.5` 不变，`fetch_cupo` 要靠它们下载 cupo 运行时。插件包自己的版本号写在 `plugins/openmm/VERSION`。

## 许可

许可和 NVIDIA cuPotential 0.3.5 相同，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。新增的能量最小化代码参照 OpenMM 8.4.0 的 `LocalEnergyMinimizer.cpp`（MIT），并带有 OpenMM 自带的 libLBFGS 头文件（MIT），两者都已列入 `NOTICE`。
