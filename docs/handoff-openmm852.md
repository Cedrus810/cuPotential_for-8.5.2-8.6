# 交接文档:OpenMM 8.4.0 → 8.5.2 迁移(截至 2026-11-22)

> **读我之前**:本文件是给下一段会话/接手人的操作交接。设计蓝图与差异审计定案在 `docs/migration-openmm852-framework.md`(下称"蓝图",含各 Phase 状态行);本文件只讲:现在到哪了、下一步干什么、哪些坑别踩。若听到"继续工作":先读蓝图 §0/§5 状态行与本文件,再向用户确认构建场地或方向,不要自行启动构建/git/环境操作。

## 0. 红线(任何接手会话,最高优先级)

1. 全局规矩(见 `~/.zcode/AGENTS.md`)始终有效:未经用户**当次**明确允许,不做任何 git 操作(含只读 status/log);禁止动环境(装包、改 site-packages、持久化环境变量等),动之前必须问。
2. 用户已拍板的现场决策,**不要重新问、更不要反着做**:
   - 2026-11-20:**暂不构建**,先做仓库内收尾(已完成);
   - **版本号(D4)暂不动**,留到 Phase 6;
   - **范围(用户 2026-11-23 原话)**:"8.4 是官方库的 。。。。 我们只是给 8.6 做适配"。即只做 8.6 线;8.4 双版本 configure、8.5.2 线编译验证不再列为待办(下表标"范围外");
   - 构建不用 docker(用户 2026-11-22 原话:"别 用 docker"、"直接 fulda 上 跑");
   - D2 定案:`openmm_dev`(CUDA 12.9)只做 API 参照与头文件级验证,真构建一律走 **13.4 Docker 线**(cupo 无 12.9 变体)。
3. 根/插件 README 首图 alt 文本里的 "OpenMM 8.4.0" 是 **8.4.0 上实测的基准数据**表述——不重跑基准不许改。
4. 工作区改动全部**未提交**(分支 main);commit 本身也未获授权,等用户点名。

## 1. 一句话状态

源码侧全部完成(零源码适配,审计级别 A),三块新逻辑(CMake 探针、系列归一正则链、Python 加载门)已离线验证,抓出并修掉 1 个真 bug(探针失败分支 `ERROR_QUIET` 吞诊断文本)。**双版本 configure、编译、打包链、GPU 回归均未执行**——一切堵在"哪台机器跑 13.4 Docker 构建"这一件事上。

## 2. 工作区清单(全部未提交)

| 文件 | 内容 |
|---|---|
| `Dockerfile` | `:82` `openmm=8.5.2`(cuda-version 仍走 `.cuda-version`=13.4.3) |
| `NOTICE`、`THIRD-PARTY.txt`(根目录和 `plugins/openmm/` 各一份,四份原本相同) | 2026-11-24:第 4 节加 `minimize/HostMinimizeKernel.cpp`(按 8.4.0 `LocalEnergyMinimizer.cpp`,MIT)与 `minimize/lbfgs.h`(libLBFGS,MIT)两条;四份同步,`cmp` 确认相同 |
| `plugins/openmm/CMakeLists.txt` | **OpenMM 系列探针**:对 `OPENMM_CORE_LIBRARY` 编译+运行一行程序实测 `getOpenMMVersion()`,归一系列(白名单 `8.4;8.5;8.6`,`-DOPENMM_SERIES=` 可跳过),产出 `NV_OPENMM_SERIES_MIN/MAX` 边界与 `NV_OPENMM_GE_85/86` 宏;2026-11-22 删掉探针编译失败分支的 `ERROR_QUIET`(与 `ERROR_VARIABLE` 同给时吞掉错误文本,实证后移除) |
| `plugins/openmm/cpp/CMakeLists.txt` | 系列宏 `PRIVATE` 挂 `OpenMMNvidia` 与 `nvplugin`(SWIG 包装同样编译 nvopenmm 头,三目标必须看到同一类形状) |
| `plugins/openmm/cpp/platforms/cuda/CMakeLists.txt` | 宏挂 `OpenMMNvidiaCUDA`;configure 期把 `cuda_min/openmm_min/openmm_max` 写入 `build/plugin_build.json` |
| `plugins/openmm/cpp/swig/NvPlugin.i` | 删 5 处 `fragment="Py_StripOpenMMUnits"` 声明(8.5 把该函数改为 `%header` 无条件注入;8.4 下靠 OpenMM 自有 double typemap 的声明仍可达)+ 说明注释 |
| `plugins/openmm/python/nv_openmm_plugin/__init__.py` | `_assert_openmm_series` 加载期版本门(D5):运行中 OpenMM 出界即拒载;dev 版本号(`8.5.2.dev-…`)可解析;元数据缺失/损坏静默放行(soft-import 契约) |
| `plugins/openmm/recipe/meta.yaml` | conda 钉扎改读包内 `runtime_build.json`(探针产物),不再手写字面量 |
| `plugins/openmm/write_package_metadata.py` | 读 `plugin_build.json`(缺 `openmm_min/max` 即报错),生成 `runtime-requirements.txt`(wheel 侧 `setup.py`/`pyproject.toml` 消费) |
| `plugins/openmm/.gitignore`(新) | `__pycache__/`(随仓库每目录局部 ignore 惯例) |
| `plugins/openmm/CMakeLists.txt`(探针,2026-11-23) | 修 bug:`execute_process` 里 `-I"${OPENMM_INCLUDE_DIR}"` 这类写法,CMake 按 legacy unquoted 参数**保留引号原样**传给编译器(`cmake -P` 实测输出 `[-I"/x/include"]`),fulda 真实 configure 报 `openmm/Platform.h: No such file`;改为整参加引号 `"-I${…}" "-L${…}" "-Wl,-rpath,${…}"`。§3 里"探针端到端 ✓"当时是在 shell 里手敲命令,shell 会剥引号,所以没覆盖 CMake 路径 |
| `plugins/openmm/cpp/platforms/cuda/src/minimize/`(新,2026-11-23,用户选 B') | `HostMinimizeKernel.{hpp,cpp}` + 原样拷贝的 `lbfgs.h`(OpenMM `libraries/lbfgs`,8.4.0 与 8.6.1 相同;实现用 `libOpenMM.so` 导出的 C 符号)。插件在 CUDA 平台登记 OpenMM 8.5+ 的 `Minimize` 内核(`CudaKernelFactory.cpp`,`#ifdef NV_OPENMM_GE_85`;8.4 构建时 `createMinimizeKernel` 恒返回 nullptr):System 里**没有** `NvOpenMM::NonbondedForce`/`MonteCarloBarostat` → 交回 stock `CommonMinimizeKernel`;**有** → 按 8.4.0 `LocalEnergyMinimizer.cpp` 在主机上跑 L-BFGS,约束弹簧在主机上用 double 加,收敛判据按 8.5+(`|g| <= tolerance*sqrt(N)`),体系力分量 ≥2e9 时抛带说明的异常(不再转 CPU)。`platforms/cuda/CMakeLists.txt` 组件列表加 `minimize` |
| `README.md` / `README.NVIDIA.md` / `README.ja.md` / `README.zh.md`(2026-11-24,用户选 A) | NVIDIA 原根 README 改名为 `README.NVIDIA.md`(内容未动,图与 "OpenMM 8.4.0" 图注随原版留着);新根 `README.md`(英)+ 日、中两份:注明非 NVIDIA 发布、与 0.3.5 的区别、依赖、安装、已验证、构建、许可。`plugins/openmm/README.md` 不动(setup.py 用它做 wheel 的 long_description)。英文 release notes:`plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md`(用户 2026-11-24:"release note 写 英语 吧") |
| `plugins/openmm/run_build` | 2026-11-22:`OPENMM=/opt/openmm` 改为 `OPENMM="${OPENMM:-/opt/openmm}"`,用于不走 docker、直接在 fulda 上用 conda env 构建(用户 2026-11-22:"别 用 docker"、"直接 fulda 上 跑");不设变量时行为不变 |
| `docs/migration-openmm852-framework.md`(新) | 蓝图 + 状态行(§2.4 施工更新、Phase 1/2/3/5/6 状态、§9 拍板状态) |
| `docs/handoff-openmm852.md`(新) | 本文件 |

钉扎机制一句话:**探针实测什么系列,`plugin_build.json` → `meta.yaml` / `runtime-requirements.txt` / Python 加载门三处消费方就钉什么边界**(`>=X.Y,<X.Y+1`);换系列(含 8.6)零手改。

## 3. 已验证 ✓ / 未验证 ✗

| 项 | 状态 |
|---|---|
| Phase 0 差异审计(8.4↔8.5.2↔8.6.1 源码 diff、SWIG 实测、conda/cupo 变体) | ✓ 蓝图 §3 定案:零适配,级别 A |
| 探针端到端(原样命令对 openmm_dev 8.5.2 编译运行) | ✓ 2026-11-22,输出干净 `8.5.2` |
| 系列归一正则链(8.5.2.dev→`8.5..<8.6`、8.6.1→`8.6..<8.7`、垃圾输入 fatal、`9.0` 归一后由白名单另层拦) | ✓ 2026-11-22,独立 cmake 脚本 7 组输入 |
| `_assert_openmm_series` 行为(界内/界外/dev/边界补零/元数据缺失与损坏) | ✓ 2026-11-22,9 例全过 |
| **8.6.1 线 configure + 全量编译(fulda,无 docker)** | ✓ 2026-11-23:env `openmm_8.6_dev`(8.6.1/py3.14,用户装了 swig 4.5.1、cmake 4.4.4)+ 系统 CUDA 13.4.92;探针输出 `8.6 (8.6.1); bounds 8.6..<8.7`;三个目标都编出来;`plugin_build.json`=`{cuda_min 13.4, openmm_min 8.6, openmm_max 8.7}`;`runtime-requirements.txt` 含 `openmm[cuda13]>=8.6,<8.7`。72 条 warning 全是 CUDA 13 的 `double4` deprecated(OpenMM 头和插件 `integrator/*.cuh` 都有)。脚本与日志:`experiments/2026-11-22_openmm86_fulda_build/`。耗时:configure 377.5 s + generate 69.1 s,run_build 合计约 51 min(NFS 上)。**未做**:import、GPU 冒烟、8.5.2 线编译 |
| 8.6.1 线 Phase 4 冒烟(fulda 5090) | ✓ 2026-11-23,冒烟通过,不是正确性验证:import、插件加载失败 0 条、bonded 能量 CUDA 对 Reference 相对差 ≤1.4e-6、nv 三件套 Context 跑 500 步、checkpoint 读回一致(`experiments/2026-11-22_openmm86_fulda_build/smoke_phase4.*`)。体系没有约束 |
| 8.6.1 线 Phase 5 单元测试(fulda,pytest `-m "not benchmark"`) | 2026-11-23 跑完:1475 passed / 15 failed / 48 errors / 13 skipped(`experiments/2026-11-23_openmm86_phase5_unit/`)。① 48 errors = `test_barostat_checkpoint` 整个文件在 setup 阶段报错,原因是 `conftest.py:134` `_assert_gpu_exclusive` 里的 nvidia-smi 首次调用超过 5 s,测试本身没跑,待补跑;② 3 failed = `test_leak_detection` 缺 cupy(环境里没装);③ **12 failed = `test_npt` 5 / `test_nv_barostat` 5 / `test_reorder` 2,在 CUDA Context 上 `LocalEnergyMinimizer.minimize` 抛 "Specified a Platform for a Context which does not support all required kernels"**,**根因已确认(2026-11-23)**:OpenMM 8.5 起 `CommonMinimizeKernel`(8.6.1 `CommonMinimizeKernel.cpp:386-438`、`kernels/minimize.cc:getConstraintEnergyForces`)把约束弹簧力加进 GPU 定点力缓冲;弹簧系数逐轮 ×10,到 k=1e10 那轮力超出定点范围(`convertForces` 阈值 2^62/2^32),于是 `evaluateCpu` 用**整个 System** 在 CPU/Reference 平台新建 Context,插件的 NonbondedForce/barostat 在这两个平台没有内核,于是抛错。证据:① 无约束同体系 minimize 成功;② MinimizationReporter 逐轮能量 stock 与插件一致到 k=1e9 轮末,插件在 k=1e10 轮开头失败(`diag_minimize2.log`);③ gdb 断点证实**全 stock 体系**在同一处也调用了 `CommonMinimizeKernel::evaluateCpu`(stock 有 CPU 内核所以不报错)。8.4.0 的 `LocalEnergyMinimizer.cpp` 在主机端加约束力,只对体系力查 2e9 阈值(读源码得出,8.4 未实跑)。修法:用户 2026-11-23 选 B'(见 §2 `minimize/`)。**修后复跑(2026-11-23,fulda,`experiments/2026-11-23_openmm86_phase5_unit/retest_minimize/`)**:增量构建通过;`diag_minimize.py` 原来失败的两组(插件力 + 约束,eager/graph)现在 minimize 成功,5 步后 PE -1.0763e+05,与 stock 同值;`test_npt` 14 passed、`test_nv_barostat` 7 passed、`test_reorder` 20 passed;另外两个也调 minimize 的文件 `test_integrator_vs_stock` 15 passed、`test_parameter_offsets` 35 passed;`test_barostat_checkpoint` 补跑 48 passed。仍未过:`test_leak_detection` 3 项(缺 cupy,环境问题) |
| 双版本 configure(8.4 + 8.5.2) | 范围外(用户 2026-11-23:只做 8.6) |
| 8.5.2 全量编译三目标 | 范围外(用户 2026-11-23:只做 8.6) |
| `run_build` 全链 + wheel/conda 候选 + 干净环境安装 | 8.6.1 线候选 wheel ✓ 2026-11-24(fulda,`pip wheel --no-build-isolation --no-deps`,不用 docker;`experiments/2026-11-24_openmm86_wheel/`):`nv_openmm_plugin-0.3.5-cp314-cp314-linux_x86_64.whl`,599 MB(其中 libcupo.so.0.3 未压缩 667 MB),30 个文件,`Requires-Dist: openmm[cuda13]<8.7,>=8.6` 及 nvidia-* `>=13.4,<13.5`(cufft `>=12.4,<12.5`),NOTICE 已含 minimize/ 两条。版本号:用户 2026-11-24 定为 `0.3.5+openmm86`(原话:"0.3.5+openmm86 这个你负责打包好"),写在 `plugins/openmm/VERSION`(根 `VERSION`/`RELEASE_TAG` 不动,fetch_cupo 要用);`plugins/openmm/CMakeLists.txt` 读 VERSION 时去掉 `+…` 再给 `project(VERSION)`(否则 configure 报 "format invalid")。正式 wheel:`dist/nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl`,599248028 字节,sha256 `c972481c489472d0449cfaa656ab2e303b7f16306246853c0607935dc48f4662`,打包时 `GITHUB_RELEASE_TAG=v0.3.5`(README 链接指官方 v0.3.5)。**干净环境安装 ✓**(用户 2026-11-24 批准建临时环境;fulda 本地 venv `/home/cyanus/omm86_wheeltest`,解释器借 openmm_8.6_dev 的 python3.14,无 LD_LIBRARY_PATH/PYTHONPATH):pip 从 PyPI 解析出 openmm 8.6.1 + OpenMM-CUDA-13 8.6.1 + nvidia cu13 13.4.92/cufft 12.4.0.43;运行时 libcudart/cufft/nvrtc/nvJitLink 全来自 venv 的 `nvidia/cu13/lib`;import 加载失败 0 条、`solvated_systems` PASS、`CUPO_TEST_INSTALLED=1` test_unit_conversion 14 passed、test_npt 14 passed(`experiments/2026-11-24_openmm86_wheel/install_test/`);测试 venv 已按用户要求于 2026-11-24 删除)。给实验室的说明:`plugins/openmm/docs/INSTALL-openmm86.md`。conda 包:✗ 未做 |
| `nm -D` 导出符号 diff(8.4.0 vs 8.5.2 构建库) | 范围外(用户 2026-11-23:只做 8.6) |
| GPU 冒烟(bonded 接管无 `std::bad_cast`;R8 栈平衡观察) | 8.6.1 线已做,见上 |
| 8.6.1 线 samples + benchmark(fulda 5090,2026-11-23) | `solvated_systems` PASS(NVT 加速比 apoa1 2.77x / bace 3.13x / dhfr 3.05x);`alchemical` PASS(charge/softcore 实时更新与 stock 一致,200 步 NVT)。benchmark 按用户要求**不锁频**(用户 2026-11-23 原话:"别调整频率 把频率 那个锁去了";cyanus 无免密 sudo,`-lgc/-rgc` 本来也执行不了;`CUPO_ALLOW_UNLOCKED_CLOCKS=1`),所以 400 项全是 "measured, not compared" skip,数字在 `experiments/2026-11-23_openmm86_phase5_bench/bench_summary.txt`:插件 vs stock 8.6.1 同卡同次运行,每步加速比 2.32x(tyk2_rbfe_dodec NVT single)~3.81x(tyk2 NPT mixed),力+能量加速比 1.13x~1.32x。cellulose、dhfr_tip4p 没有出现在结果里(推测数据不在树里,未核实)。没有 8.4 基线可比 |

## 4. 下一步(按序;先问用户场地)

**A. 构建场地**(有 docker 权限的机器;当前登录节点 docker.sock 被拒、无调度器、module 无 cuda/13.4):
```bash
cd <repo>
./run_docker plugins/openmm/run_build    # 镜像自动按 8.5.2 建;run_build 会先 fetch_cupo 取 13.4 变体
```
- Phase 1 验收还要求**同树对 8.4 再 configure 一次**(单源码双版本证明;`-DOPENMM_*` 指到 8.4 安装,或用 8.4 镜像)。
- configure 前置:libcupo.so 必须在位(缺失即 FATAL);**别在 openmm_dev 里试全量构建**(12.9 无 cupo 变体,D2)。
- Phase 3 验收:run_build 全链 → 候选包;`nm -D` 对比;干净环境 `import nv_openmm_plugin` 三种安装形态(commands.md 在 `plugins/openmm/docs/commands.md`)。

**B. GPU 节点**(计算节点提交;清单见蓝图 Phase 4/5):`./run_test openmm_unit`(141 项)→ vs_stock 对拍 → golden → `test_tyk2_alchemical`/`test_npt`/`test_nv_barostat` → 序列化/重启 → `./run_test openmm_bench`。任何失败禁止调容差放行,必须归因。

**C. Phase 6 收口**(Phase 5 全绿后才动,现在动会撒谎):README/commands.md 支持性表述("requires OpenMM 8.5.x");`_benchmark_systems.py` 的 `OPENMM_RELEASE`→8.5.2(哈希已实测跨 tag 相同,纯表述);`VERSION`/`RELEASE_TAG`/release notes;D4 版本号届时问用户。

**D. Phase 7(8.6 线)**:源码零改动(§3.2c 定案),探针自动出 `8.6..<8.7`;同流程重跑;是否提前合并(D3)由用户在 8.5.2 线全绿后拍板。可选项(独立功能,非迁移):Nv barostat 跟进 8.6 逐粒子缩放。

## 5. 坑与事实(别再查一遍)

- **登录节点**:无 GPU、无 docker 权限、无 sbatch/qsub、module cuda 12.4–13.3(无 13.4)、podman 缺 subuid、apptainer 1.5.0 在(未走通,非标准路径)。
- **openmm_dev**(`/home/ruigengji/miniforge3/envs/openmm_dev`):OpenMM 8.5.2 + swig 4.5.0 + CUDA 12.9;只读参照用,不装东西。
- `_benchmark_systems.py` 下载的 apoa1.pdb / 5dfr_solv-cube_equil.pdb 在 8.4.0 与 8.5.2 tag 下 **SHA-256 相同**(2026-11-20 下载实测)。
- conda-forge 8.5.2 / 8.6.0 / 8.6.1 均有 CUDA 13 系 py312 变体(8.6.1 有显式 `>=13.4,<14` 档);cupo v0.3.5 release 只有 CUDA 13.4 tarball。
- **R9 规约**:今后序列化代码对 `SerializationNode::getStringProperty/getProperties` 只按值接收(8.6 起按值返回,取指针/跨作用域存引用会悬垂);现有唯一绑定 `NonbondedForceProxy.cpp:149` 已核安全。
- 全仓已 grep:plugins/openmm 之外无遗留 8.4 钉扎;README 图注、PROVENANCE、史实注释里的 8.4 字样为**合法保留**。
- 全部 /tmp 审计与测试产物已清理,仓库外无遗留。

## 6. 决策点状态(蓝图 §9)

| # | 状态 |
|---|---|
| D1 源码组织 | (a) 单源码+宏,已落地(零调用点,纯保险) |
| D2 openmm_dev 角色 | (a) 定案:仅 API 参照 |
| D3 钉扎策略 | (a) 系列封闭,探针自动推导;8.6 提前合并留口子 |
| D4 版本号 | **未拍板**(用户 2026-11-20:暂不动),Phase 6 再问 |
| D5 加载期断言 | 已落地,2026-11-22 行为测试通过 |
| — 构建场地 | 待用户安排(当前唯一硬阻塞) |

