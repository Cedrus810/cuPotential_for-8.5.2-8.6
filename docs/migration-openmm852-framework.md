# cuPotential OpenMM 插件迁移设计框架:OpenMM 8.4.0 → 8.5.2(→ 8.6)

> **文档性质:施工蓝图,不是施工记录。** 本文档只定义"怎么迁",不含任何对仓库/环境的改动。
> 撰写时点:2026-11-20。§3 的源码级审计基于 GitHub 官方 tag tarball 对比与本地 SWIG 实测完成(方法与产物见附录 B);全程未执行 git 操作、未改动环境、未修改仓库源码(期间误产的一个 SWIG 生成文件已当场清除)。
> **施工进展(2026-11-20)**:Phase 1-3 的**源码侧**已落地——钉扎改为探测驱动单一来源(§2.4 施工更新),各阶段状态行见 §5。**双版本 configure、编译、打包链均未执行**:本账号在当前登录节点无 docker 权限(docker.sock permission denied)、无调度器命令、module 无 cuda/13.4;按 D2 真构建走 13.4 Docker 线,场地待用户安排(当日拍板:暂缓构建,先做仓库内收尾)。
> **交接文档:`docs/handoff-openmm852.md`**(截至 2026-11-22,含下一步命令与红线清单;新会话从它进入)。

---

## 0. 一页结论

| 项 | 结论 |
|---|---|
| 迁移命题 | 插件从 **OpenMM 8.4.0** 迁到 **OpenMM 8.5.2**(环境 `openmm_dev`,已装 8.5.2);同一框架复用于后续 8.6 |
| CUDA 线 | 按用户拍板:**按 13.4 生产线迁移**(`.cuda-version`=13.4.3,Docker 链不变);`openmm_dev` 的 CUDA 12.9 仅作为开发/联调环境,见 §9 决策点 D2 |
| 兼容性判断(已定案,证据见 §3) | **编译层无断点,8.5.2 与 8.6 均已审计**:①8.4.0→8.5.2 对插件耦合面(API 头、`internal`/`common`/`cuda` 平台头、bonded 内核三接口、`PlatformData::contexts`)**零修改、纯新增**;②8.5.2→8.6.1 仅三处实质变更且**均不触及插件**(stock barostat 内核加剪切尾参、`SerializationNode` 两个 getter 改按值返回——插件用法已核安全、PythonForce 内部扩展);SWIG 层用真实 `NvPlugin.i` + swig 4.5.0 + 8.5.2 头实测**通过**(仅良性 Warning 490);conda-forge 8.5.2/8.6.0/8.6.1 均有 CUDA 13 系 py312 变体,Docker 生产线 `cuda-version=13.4` 三个版本都可直接解析 |
| 最大的单点风险 | 插件**接管了 OpenMM 原生 bonded 内核工厂**(HarmonicBond/Angle/PeriodicTorsion,`CudaKernelFactory.cpp:53-55`)——8.5.2 下接口未变,风险降级为"运行期行为回归"而非编译断点;另一观察项是 8.5.2 新增的 `popAsCurrent()` 栈平衡校验(插件 RAII `ContextSelector` 用法,见 §3.2b/R8) |
| 工作量主形态 | **重编译 + 三处版本钉扎 bump + 全量回归**;预期**零源码适配**(兼容宏机制保留但预计不用,见 §4.1),以 GPU 节点上的 Phase 4/5 实测为最终裁决 |

---

## 1. 目标与非目标

### 1.1 目标
1. `nv-openmm-plugin` 0.3.x 源码可针对 **OpenMM 8.5.2** 构建、加载、通过现有测试套件(单元 + golden + 对拍),并在 8.5.2 上保持与 8.4.0 等价的数值容差与性能水位。
2. 发布链(conda recipe、wheel 元数据、Docker、CI)的 OpenMM 钉扎从 8.4 系升到 8.5 系,三处联动、不漏一处。
3. 迁移方式**单源码兼容 8.4 与 8.5.2**(理想情形,见决策点 D1),为 8.6 再迁移留下同一套框架与流程。

### 1.2 非目标
- 不改 cuPotential 运行时(`libcupo.so`,ABI 32)的内部实现;只处理它与工具链/OpenMM 的**接口面**。
- 不动 CUDA 13.4 生产线布局;不在本迁移中引入新硬件/新架构支持。
- 不做 8.6 的实际施工;8.6 仅作为框架预留(§7 Phase 7)。
- 不改公共 Python API(`nv_openmm_plugin` 的用户接口)语义。

---

## 2. 现状基线(已核实事实)

### 2.1 版本矩阵

| 组件 | 当前发布链(8.4 线) | 迁移目标 | 开发/联调环境 `openmm_dev` |
|---|---|---|---|
| OpenMM | 8.4.0(`Dockerfile:82` 钉死) | **8.5.2** | 8.5.2(conda-forge, `py312h5a97af1_0`) |
| CUDA | 13.4.3(`.cuda-version`) | 13.4(不变) | **12.9**(cuda-version 12.9, nvcc 12.9.86, cufft 11.4.1.4) |
| swig | conda-forge ≥4.3(Docker 内 `swig`) | 与目标 OpenMM 的 `_openmm.so` runtime 匹配 | 4.5.0 |
| Python | 3.12 | 3.12(不变) | 3.12.13 |
| 插件版本 | 0.3.5(`VERSION`) | 0.3.x 或 0.4(见 D4) | — |
| cupo 运行时 | `libcupo.so`,CUPO_ABI_VERSION=32,按 CUDA 版本出 tarball(`fetch_cupo` 从 release SHA256SUMS 按 tarball 名选 CUDA 变体) | 不变 | 需按所用工具链取对应变体(见 D2) |

### 2.2 插件组成(迁移对象)
- **三个自研 OpenMM 类**(`plugins/openmm/cpp/openmmapi/`):`NvOpenMM::NonbondedForce`(含 Custom 模式)、`NvOpenMM::LangevinMiddleIntegrator`、`NvOpenMM::MonteCarloBarostat`,各带 ForceImpl 与序列化 Proxy。
- **CUDA 平台实现**(`cpp/platforms/cuda/`,产出 `libOpenMMNvidiaCUDA.so`):nonbonded / integrator / barostat / bonded 四组内核 + kernel factory。
- **grouped bonded 适配器**(`bonded/GroupedBonded.cu`):实现 **OpenMM 原生** `CalcHarmonicBondForceKernel`/`CalcHarmonicAngleForceKernel`/`CalcPeriodicTorsionForceKernel` 接口并在 CUDA 平台上**顶替原生工厂**(`CudaKernelFactory.cpp:53-55` 注册原名内核),内部带 stock 回退路径。
- **SWIG 层**(`cpp/swig/NvPlugin.i` → `_nvplugin.so`):依赖 OpenMM 安装的 `include/swig/typemaps.i` 与 `swig/OpenMMSwigHeaders.i` 做跨模块类型共享(让 `NvLangevinMiddleIntegrator` 成为 `openmm.Integrator` 同一 SWIG 类型体系的子类)。
- **Python 包装**(`python/nv_openmm_plugin/__init__.py`):固定加载序(libOpenMMNvidia 先 RTLD_GLOBAL → nvplugin → loadPluginsFromDirectory)、cuFFT JIT 工具链预载、RNG/确定性开关。

### 2.3 与 OpenMM 的耦合面(按风险分层)

| 层 | 内容 | 风险 | 说明 |
|---|---|---|---|
| L1 公共 API | `openmm/{System,Context,Kernel,Platform,Force,Integrator,Vec3,各种 Force}.h`、`serialization/*`、`kernels.h` | 低 | OpenMM 跨版本稳定承诺范围内 |
| L2 内部 API 头 | `openmm/internal/{ContextImpl,ForceImpl,NonbondedForceImpl,windowsExport,AssertionUtilities}.h`、`reference/SimTKOpenMM*.h` | 中 | 无兼容承诺,但多年少动 |
| L3 平台内部头 | `openmm/cuda/{CudaContext,CudaPlatform,CudaArray,...}.h`、`openmm/common/{ComputeContext,ComputeArray,ComputeVectorTypes,ComputeForceInfo,ContextSelector,NonbondedUtilities,IntegrationUtilities}.h`;源码树构建时还加 `olla/include`、`libraries/lepton`、`platforms/{common,cuda}` 路径(`plugins/openmm/CMakeLists.txt:124-140`) | **高** | 8.5 明确带"下一代平台支撑"改动,此处最可能变 |
| L4 SWIG 层 | `include/swig/{typemaps,OpenMMSwigHeaders}.i`;runtime data 版本必须与目标 `_openmm.so` 一致(现为 swig_runtime_data5) | 中高 | swig 4.x 内 runtime 同版本;跨大版本必炸 |
| L5 构建系统 | `run_build:79-89` 手工传 `OPENMM_INCLUDE_DIR / OPENMM_CORE_LIBRARY / OPENMM_CUDA_LIBRARY / OPENMM_EXTRA_INCLUDE_DIRS`;CMake 无 `find_package(OpenMM)` 版本约束 | 中 | OpenMM 安装布局若变(如 pip 安装形态),此链需适配 |

### 2.4 版本钉扎位置(必须三处联动)
1. `plugins/openmm/recipe/meta.yaml:67` — `openmm >=8.4,<8.5.0a0`(注释明确说 bump 时与下条同动;并记录了"裸 `>=8.4` 会静默解析到 8.6.0"的教训:`libOpenMM.so` 无版本 soname,加载器不拦 ABI 错配)。
2. `plugins/openmm/write_package_metadata.py:43` — `openmm[cuda{N}]>=8.4,<8.5`(wheel 的 runtime-requirements.txt)。
3. `Dockerfile:82` — `micromamba create ... openmm=8.4.0 ...`(构建与 CI 环境)。
另:README/文档中的"OpenMM 8.4.0"表述(根 README、插件 README 图注)随版本更新。

> **[施工更新 2026-11-20]** 三处手改联动已按更稳的形态落地:**探测驱动单一来源**。CMake 探针(`plugins/openmm/CMakeLists.txt:109-192`)对 `OPENMM_CORE_LIBRARY` 编译并运行一行程序实测 `Platform::getOpenMMVersion()`,归一到系列(白名单 `NV_OPENMM_ALLOWED_SERIES` = 8.4/8.5/8.6;`-DOPENMM_SERIES=` 可跳过探针),推导 `NV_OPENMM_SERIES_MIN/MAX` 与 `NV_OPENMM_GE_85/86` 宏,写入 `build/plugin_build.json`。三个消费方——conda recipe(`meta.yaml` 读包内 `runtime_build.json`)、wheel requirements(`write_package_metadata.py` 读 `plugin_build.json` 生成 `runtime-requirements.txt`)、Python 加载门(`__init__.py::_assert_openmm_series`,即 D5)——读的都是这份产物,重建即重推导,不再逐发布手改。上列字面量位置保留作历史索引;`Dockerfile:82` 已改 8.5.2。README 图注一条的更正见 Phase 6 状态行。

---

## 3. 差异分析(8.4.0 → 8.5.2)

### 3.1 官方 release notes(已核实,GitHub releases 原文)

**8.5.0**(2026-03-20)新特性:
- 能量最小化器重写为全 GPU 执行(新增 `MinimizeKernel` 纯虚内核类,各平台须实现;插件不注册该名,由 stock CUDA 工厂提供)。
- 新增 `PythonForce`(Python 代码算力,新增 `CalcPythonForceKernel`)。
- 新增 LCPO 隐式溶剂表面积方法(新增 `CalcLCPOForceKernel`)。
- "为下一代 OpenFF 力场做的内部改动"(release notes 原文)——经 diff 核实,**未改变插件耦合的既有接口**(见 §3.2b)。
- 新增 `Platform::getDevices()` 设备查询 API;CUDA/OpenCL 平台各自实现。
- Gromacs 读入多支持一种 vsite。

**8.5.1**(2026-04-10)修复:minimizer 能量比较精度、worker 线程多力并发、FlexibleBarostat 压强计算。

**8.5.2**(2026-06-08)单条修复:**#5279 "Make sure contexts are deselected before evaluation"**——新增 `ContextDeselector` RAII 类,并在 `popAsCurrent()` 里加了"弹出的 context 不是预期值则抛 OpenMMException"的栈平衡校验(对插件的影响见 §3.2b)。

**8.6.0**(2026-08-19):multistate 采样器(ReplicaExchangeSampler/ExpandedEnsembleSampler);`CustomNonbondedForce` 长程校正系数缓存;PythonForce 支持粒子子集;MonteCarloBarostat/Anisotropic/Membrane 增加逐粒子独立缩放选项(默认仍为分子整体缩放)。
**8.6.1**(2026-09-11):PythonForce 段错误修复、CustomNonbondedForce 表函数更新修复等。

> 对插件的净判断:8.5.x 的变更**全部是加法或与插件无关的重构**;唯一触碰"执行语义"的是 8.5.2 的 context 栈校验,属观察项非适配项。8.6 系的源码级审计见 §3.2c,结论同样是零适配。

### 3.2 已核实项(用 `openmm_dev` 的 OpenMM 8.5.2 安装逐项对照,2026-11-20)
1. **头文件布局齐全**:`include/openmm/{cuda,common,internal,reference}`、`include/swig/{typemaps.i,OpenMMSwigHeaders.i}`、`include/lepton` 均存在,与 `run_build` 传入的路径假设一致。
2. **插件使用的内部 API 成员全部在位**(逐名 grep 核对,零缺失):
   - `CudaContext`/`ComputeContext`:`getCurrentStream/setCurrentQueue/getCurrentQueue/createQueue/getNumAtoms/getPaddedNumAtoms/getPosq/getPosqCorrection/getVelm/getLongForceBuffer/getEnergyBuffer/getForce/addForce/getAtomIndex/setAtomIndex/reorderAtoms/addReorderListener/invalidateMolecules/getPostComputations/getUseMixedPrecision/getUseDoublePrecision/getPeriodicBoxVectors/unwrap/getDeviceIndex/getIntegrationUtilities/getStepCount/setStepCount/setTime/getTime/getComputeForceCount/setComputeForceCount/getAtomsWereReordered/getPosCellOffsets/setPosCellOffsets/initializeContexts/setForcesValid`
   - `ContextImpl`:`getParameter(s)/getSystem/getPeriodicBoxVectors/setPeriodicBoxVectors/calcForcesAndEnergy/updateContextState/getPlatformData/getPlatform/getOwner/getMolecules/systemChanged/getLastForceGroups/getIntegrator/getForceImpls`
3. **`CudaPlatform::PlatformData` 仍有 `std::vector<CudaContext*> contexts`**(`cuda/CudaPlatform.h:142`)——插件 `createKernelImpl` 里的 downcast 取主 context 的模式不变。
4. **bonded 内核接口签名**:`kernels.h` 中 `copyParametersToContext(ContextImpl&, const Force&, int first, int last)`;插件 `GroupedBonded.cu:306` 已实现带 `(first,last)` 的 override——**匹配**。
5. **`ForceImpl` 兼容**:两参 `updateContextState(ContextImpl&, bool& forcesInvalid)` 仍在(插件 barostat Impl 用此签名),另存在一参重载,互不冲突。
6. SWIG 接口引用的 `OpenMM.h/OpenMMAmoeba.h/OpenMMDrude.h/DPDIntegrator.h` 均存在。
7. 新增 Impl 头(`LCPOForceImpl.h`、`PythonForceImpl.h`、`ConstantPotentialForceImpl.h` 等)只做加法,不挤占既有接口。

> 以上是**成员存在性**层面的核查;下节是 GitHub 源码 diff + 本地实测的**定案级**核查。

### 3.2b 源码 diff 审计定案(8.4.0 ↔ 8.5.2,2026-11-20 执行)

方法:GitHub 官方 tag tarball(`openmm-8.4.0/8.5.0/8.5.1/8.5.2.tar.gz`)解包后逐文件 `diff -r`,聚焦插件耦合面;并用 `openmm_dev` 的 swig 4.5.0 对真实 `NvPlugin.i` 实测(附录 B)。

**结论:插件耦合面零断点,全部变更纯新增或无关。**

| 文件(8.5.2 相对 8.4.0) | 变更内容 | 对插件 |
|---|---|---|
| `olla/include/openmm/kernels.h` | ① `UpdateStateDataKernel::getPositions` 加 `bool allowPeriodic=false` 尾参;② `CalcCustomCPPForceKernel::initialize(System&→ContextImpl&)`;③ 新增 `MinimizeKernel`/`CalcLCPOForceKernel`/`CalcPythonForceForce` 三个纯虚类 | 无影响——插件不实现 UpdateStateData/CustomCPP;**bonded 三内核类(HarmonicBond/Angle/PeriodicTorsion)逐字未动**;新增类由 stock CUDA 工厂按名提供 |
| `openmmapi/include/openmm/internal/ContextImpl.h` | 新增 `minimize()`、`getPositions` 尾参(带默认值)、私有成员 `minimizeKernel/hasMinimizeKernel` | 无影响(插件调用的方法全部未动) |
| `openmmapi/include/openmm/NonbondedForce.h` | 仅注释错字("oaram"→"param") | 无影响(插件转换构造函数所依赖的 stock API 原样) |
| `openmmapi/include/openmm/Context.h` / `State.h` / `CMMotionRemover.h` | `friend class LocalEnergyMinimizer`;State 私有成员加 `hasBoxVectors`;CMMotionRemover 仅文档 | 无影响 |
| `platforms/cuda/include/CudaContext.h`(装为 `openmm/cuda/CudaContext.h`) | 新增 `static ensureCudaInitialized()` | 无影响 |
| `platforms/cuda/include/CudaPlatform.h` | 新增 `getDevices()` | 无影响(`PlatformData::contexts` 等既有成员未动) |
| `platforms/common/include/openmm/common/ContextSelector.h` | 新增 `ContextDeselector` RAII 类(8.5.2 #5279) | 新增类,插件可忽略;行为影响见下 |
| `platforms/common/include/openmm/common/CommonKernels.h` | 新增 `CommonCalcLCPOForceKernel`;`getPositions` 尾参 | 无影响 |
| `wrappers/python/.../swig_lib/python/typemaps.i`(装为 `include/swig/typemaps.i`) | `Py_StripOpenMMUnits` 从 `%fragment("Py_StripOpenMMUnits","header")` 改为 `%header { }` 内联;相关 typemap 去掉 fragment 声明 | **Warning 490 良性**:函数改经 `%header` 无条件注入每个生成文件,fragment 引用悬空只产生警告。已用真实 `NvPlugin.i` 实测:swig 4.5.0 + 8.5.2 头 **exit=0**,生成的包装代码中函数在第 3380 行注入可用。可选清理:删掉 `NvPlugin.i` 里 5 处 `fragment="Py_StripOpenMMUnits"` 声明消警告(非必须) |
| `platforms/cuda/src/CudaKernelFactory.cpp` | 新增 Minimize/PythonForce/LCPO 三名注册 | 无影响(与插件按名注册并存,键不冲突) |
| `platforms/cuda/src/CudaContext.cpp` | `popAsCurrent()` 加校验:弹出 context ≠ 预期即抛 `OpenMMException`;新增 ensureCudaInitialized | **行为观察项 R8**:插件在 barostat/bonded 用 RAII `ContextSelector`(push/pop 配对使用),正常用法不受影响;8.5.2 只是把"栈不平衡静默损坏"变成"响错"。Phase 4 冒烟确认即可 |
| `platforms/cuda/src/CudaContext.cpp`(JIT 选项) | `defaultOptimizationOptions = "--use_fast_math"` 原样保留(行 135) | 无影响(插件 AOT 内核的 `--use_fast_math;--ftz=false` 基线不变) |
| `openmmapi/src/MonteCarloBarostatImpl.cpp` | 纯写法重构(brace-init),语义不变 | 无影响(插件自研 Impl 独立) |
| `platforms/cuda/include/CudaKernels.h`(stock CUDA bonded 内核) | **未修改** | 无影响——`GroupedBonded` 的 stock 回退路径(`GroupedBonded.cu:293` 动态 `createKernelImpl`)接口原样 |
| `serialization/include/openmm/serialization/SerializationProxy.h` | 未修改;`SerializationProxyRegistration.cpp` 仅新增 LCPO 注册 | 无影响(插件 `NvNonbondedForce` proxy 原样) |
| `platforms/reference/include/SimTKOpenMMUtilities.h` | 注释/格式清理 | 无影响(仅测试辅助驱动引用) |
| 其余(openmmapi/src/*、common.cu、findInteractingBlocks.cu 等) | 行为实现优化(GPU minimizer 等) | 属 stock 路径行为变化,由 Phase 5 的 stock-vs-stock 基线对拍吸收(R6) |

**8.5 系没有 `find_package(OpenMM)`/安装布局变化**,`run_build` 的手工路径传参模式在 8.5.2 下原样可用(conda-forge 包的 `include/openmm/{cuda,common,internal}`、`include/swig/`、`lib/plugins/` 布局与 8.4 相同,已对照 `openmm_dev` 实装核实)。

### 3.2c 源码 diff 审计定案(8.5.2 ↔ 8.6.1,2026-11-20 执行,8.6 线同样定案)

方法同 §3.2b(附录 B)。8.6.0 与 8.6.1 差异仅为其 release notes 所列修复,以下以 8.6.1 为终态。

**结论:三处实质变更,全部不触及插件;插件耦合面成员扫描零缺失。**

| 文件 | 变更内容 | 对插件 |
|---|---|---|
| `olla/include/openmm/kernels.h` | stock `ApplyMonteCarloBarostatKernel::scaleCoordinates` 增加 `scaleXY/scaleXZ/scaleYZ` 三个尾参(默认 0,上三角形变,服务全应力张量) | 无影响——插件实现的是**自研** `NvOpenMM::ApplyMonteCarloBarostatKernel`(内核名 `NvApplyMonteCarloBarostat`,见 `nvopenmm/Kernels.h:151`),不实现也不调用 stock 类;stock 的 Common 实现自行跟进 |
| `openmmapi/include/openmm/MonteCarloBarostat.h` | ctor 尾参 `bool scaleMoleculesAsRigid=true` + inline get/set(`getScaleMoleculesAsRigid` 等) | 无影响——`NvMonteCarloBarostat` 是独立类(SWIG 三参 ctor 原样);插件行为保持"分子质心刚体缩放"(其设计文档即如此)。是否跟进 8.6 的逐粒子模式属功能增强,列入 Phase 7 可选项而非迁移项 |
| `serialization/include/openmm/serialization/SerializationNode.h` | `getProperties()` 与两个 `getStringProperty` 重载**返回类型从引用改按值**;私有存储 `std::map`→`std::string`;新增 `findPropertyValue` | **重编译覆盖、源码兼容(已逐点核实)**:插件用到的 `getIntProperty/getDoubleProperty/getChildNode/getChildren/setIntProperty/setName` 签名全部未动;唯一引用绑定 `NonbondedForceProxy.cpp:149` 的 `const auto& weight = ...getStringProperty(...)`,按值返回下靠临时生命周期延长仍合法,且该引用仅在作用域内立即使用。**派生约束见 R9**:今后代码不得对该返回值取指针(`&getStringProperty(...)`)或跨作用域存引用——那在 8.5.2 安全、8.6.1 悬垂 |
| `platforms/common/include/openmm/common/CommonKernels.h` | stock barostat Common 实现签名跟进;`PythonForceComputation` 私有成员扩展(粒子子集) | 无影响(均 stock 侧,插件不继承) |
| `platforms/common/include/openmm/common/CommonCalcCustomNonbondedForceKernel.h` | 私有 LRC 缓存成员(`longRangeCoefficientCache`) | 无影响(插件不继承;插件读取 stock `CustomNonbondedForce` 定义的方式为公开 API,加法不破坏) |
| `openmmapi/include/openmm/DPDIntegrator.h` | 仅注释措辞 | 无影响(`NvPlugin.i` %import 的头之一) |
| `wrappers/python/.../exceptions.i` | 修改 | 无影响——`NvPlugin.i` 只 include `std_string.i`、`swig/typemaps.i`、`swig/OpenMMSwigHeaders.i`;**`typemaps.i` 在 8.6.1 未再改动** |
| stock 序列化(`MonteCarloBarostatProxy` 等) | stock 内部跟进 | 无影响(插件自研 `NvNonbondedForce` 等 proxy 独立) |
| 未修改项 | `ContextImpl.h`、`CudaPlatform.h`(`PlatformData::contexts` 仍在,8.6.1 行 142)、`CudaContext.h`、`Platform.h`、kernels.h 的 bonded 三内核类、`SerializationProxy.h` | — |
| 成员存在性扫描 | 插件使用的全部 `CudaContext`/`ComputeContext`/`ContextImpl` 成员(§3.2 那批)对 8.6.1 源码树逐名核对 | **零缺失** |
| conda-forge 变体 | 8.6.0/8.6.1 linux-64 py312 均有 CUDA 12.9 与 `>=13.0,<14` 系构建;8.6.1 另有显式 `>=13.4,<14` 档 | Docker 13.4 线对 8.5.2/8.6.0/8.6.1 均可直接解析 |
| 行为面(`Integrator.cpp`、stock barostat Impl/Proxy、若干 `kernels/*.cc`) | stock 路径行为优化/修复 | Phase 7 的 stock-vs-stock 基线对拍吸收(同 R6 方法) |



### 3.3 待核实清单结案(原 Phase 0 输入,2026-11-20 全部有结论)

- [x] **release notes 逐条过**(8.5.0/8.5.1/8.5.2 + 8.6 预读)——见 §3.1。
- [x] **头文件全文 diff**(`internal`/`common`/`cuda`/`swig`/olla)——见 §3.2b:插件耦合面零修改、纯新增;bonded 三内核接口逐字未动。
- [x] **swig 版本/runtime 兼容**——不以 feedstock 记录为准,直接实测:`openmm_dev` 的 swig 4.5.0 + 8.5.2 安装头 + 真实 `NvPlugin.i` → **exit=0**(仅 Warning 490/453/314,均良性;8.4 构建同样有 453/314)。跨模块 `%import(module="openmm")` 正常,SWIG runtime 同代(4.x,swig_runtime_data5)。
- [x] **conda-forge 8.5.2 的 CUDA 13 变体存在**——linux-64 同时发布 12.9 系(`cuda-version >=12.9,<13`,即 `openmm_dev` 里已装的 `py312h5a97af1_0`)与 13 系(`cuda-version >=13.0,<14`,`py312ha28768a_0` 等)构建;Docker 生产线 `cuda-version=13.4` 解析到 13 系构建,**打包链无阻塞**。
- [x] **8.6 线同标准审计**(§3.2c):8.5.2→8.6.1 三处实质变更均不触及插件;conda-forge 8.6.0/8.6.1 均有 CUDA 13 系 py312 变体(8.6.1 有显式 13.4 档),打包链同样无阻塞。
- [x] **cupo v0.3.5 release 只有 CUDA 13.4 变体**(`cupo-0.3.5+cuda_13.4.sm_75_80_86_89_90_100_103_120.ge7cfd685.tar.gz`,SHA256SUMS 在案)——**没有 12.9 tarball**。含义:`openmm_dev`(CUDA 12.9)里**无法**用已发布 cupo 运行时做完整插件构建;完整构建只能走 13.4 线(Docker/计算节点),或自建 cupo 12.9(需 cupo 源码,本仓库树不含,成本高)。已并入 D2。
- [ ] **导出符号 nm diff(8.4.0 vs 8.5.2 构建库)**——需先有两版编译产物,归入 Phase 1 验收(双版本构建成功后 `nm -D` 对比);头文件零删除 + Phase 4 运行期加载已构成实质兜底。
- [x] **8.5 "下一代平台"内部改动是否波及既有 CUDA 平台行为**——diff 核实:未动既有接口;行为面变化集中在 GPU minimizer/Python 回调;`--use_fast_math` JIT 默认、reorder 机制、`PlatformData::contexts` 均原样。

---

## 4. 总体迁移策略

1. **单源码、版本探测**:目标是一个源码树同时可对 8.4 与 8.5.2 构建(方式见 D1)。CMake 侧新增对 OpenMM 版本的显式探测(读 `openmm/version.h` 宏),据此设置兼容宏(如 `NV_OPENMM_GE_85`),让差异点用 `#if` 收口而不是分叉代码。**§3.2b 审计表明预期差异点为零——该机制作为保险而非预期开销。**
2. **8.5.2 先行,8.6 同框架复用**:Phase 0-6 针对 8.5.2;8.6 作为 Phase 7 增量,重跑同一 diff/构建/回归流程。钉扎策略保持"系内开放、跨系封顶"(8.5 线:`>=8.5,<8.6`)。
3. **CUDA 13.4 主线**:所有正式构建、发布物、CI 按 13.4(与现状一致);`openmm_dev`(12.9)用于 API 层验证与开发联调,不产出发布物。两条线不混装(soname:13.x 为 `libcudart.so.13`/`libcufft.so.12`,12.9 为 `.so.12`/`.so.11`;cupo 的嵌入式表达式 IR 与 NVRTC 版本绑定,见 `meta.yaml:52-53` 注释)。
4. **不动运行环境的原则**:一切需改动 `openmm_dev`/生产环境的动作(装包、升级 CUDA 组件)列为**独立审批节点**,逐条征得用户同意后执行;默认路径是不改环境(见 D2/D3)。
5. **防护性设计(建议项,施工时实现)**:加载时对 `Platform.getOpenMMVersion()`(或 `openmm.__version__`)与编译期版本做一致性断言,防止"无版本 soname 静默错配"再现(`meta.yaml` 注释里记录过 `>=8.4` 解析到 8.6.0 未报错的事故)。

---

## 5. 施工阶段划分

> 每阶段含:目标 → 动作 → 产出 → 验收门槛。阶段间串行,阶段内可并行。**任何阶段不改动用户环境前都必须单独获批。**

### Phase 0 — 差异审计定案(纯只读)— **已于 2026-11-20 执行,见 §3 与附录 B**
- **状态**:release notes / 头文件 diff / SWIG 实测 / conda 变体 / cupo 变体均已完成并回填 §3.1–§3.3;**工作量级别判定:A(纯重编译,预期零源码适配)**。
- **残余项**:① `nm` 导出符号 diff(需两版编译产物,移入 Phase 1 验收);② 一切 GPU 上的运行期结论(本机无 GPU,归 Phase 4/5)。
- **验收**:每一条 §3.3 项有结论(已满足);工作量级别经用户确认(待确认)。

### Phase 1 — 构建链切换准备
- **目标**:让"针对 8.5.2 的构建"可复现。
- **动作**(按批准顺序):Docker 基线 `openmm=8.4.0` → `8.5.2`(Dockerfile:82,cuda-version 维持 13.4);若需本地联调,另定 `openmm_dev` 的工具链方案(D2);CMake 增加版本探测与兼容宏骨架。
- **产出**:可在 8.5.2 头文件/库上配置成功的 CMake 工程;一份"构建环境清单"(镜像/环境、路径、变量)。
- **验收**:同一路径下,8.4 与 8.5.2 两次 configure 均成功(证明单源码双版本可行,或在此发现必须分叉的点并回报)。
- **状态(2026-11-20)**:源码侧完成——`Dockerfile:82` 已改 `8.5.2`;CMake 探测落地,形态为探测驱动单一来源(见 §2.4 施工更新)。**双版本 configure 验收未执行**(构建场地见文首施工进展)。注意:配置期即需 libcupo.so 在位(架构探测与 `Cupo::Runtime` 导入目标都 fatal 于缺失),验收前需先 `fetch_cupo` 取 13.4 变体。
- **新逻辑离线验证(2026-11-22,登录节点,零环境改动)**:①探针程序按 CMakeLists 原样命令对 `openmm_dev` 的 8.5.2 编译运行,输出干净 `8.5.2`;②系列归一化正则链独立 cmake 脚本验证(8.5.2.dev→`8.5..<8.6`、8.6.1→`8.6..<8.7`、垃圾输入走 fatal、`9.0` 由白名单另层拦截);③加载门 `_assert_openmm_series` 行为测试 9 例全过(界内/界外/dev 版本号/边界补零/元数据缺失与损坏静默);④修复探针编译失败分支的 `ERROR_QUIET` 吞错(与 `ERROR_VARIABLE` 同给时丢弃诊断文本,FATAL_ERROR 变空话;实证后删 `ERROR_QUIET`)。

### Phase 2 — 源码适配
- **目标**:针对 Phase 0 报告里的真实差异逐项适配。
- **动作**:接口签名对齐(优先 bonded 适配器与三组自研内核);SWIG 层随 8.5.2 的 `swig/` 输入重新生成 `_nvplugin.so`;序列化 Proxy 核对;`#if` 兼容宏收口所有版本差异;废弃 API 替换(若有)。
- **产出**:编译零警告(至少零错误)通过 8.5.2 的三个目标:`libOpenMMNvidia.so`、`libOpenMMNvidiaCUDA.so`、`_nvplugin.so`。
- **验收**:8.5.2 全量编译通过;回退开关有效(同码仍可对 8.4 编译,若 D1 选单源码)。
- **状态(2026-11-20)**:兼容宏骨架已挂——`NV_OPENMM_GE_85/86` 由探针按系列注入,`PRIVATE` 挂到 `OpenMMNvidia`、`OpenMMNvidiaCUDA` 与 `nvplugin`(SWIG 包装的 C++ 编译同样编译 nvopenmm 头,三目标必须看到同一类形状;SWIG 自身解析期预处理独立,首个需 SWIG 条件感知的调用点出现时再显式处理)。cpp 源码**零** `NV_OPENMM_GE*` 调用点(纯保险,与 §3.2b 零适配定案一致);源码适配为零,级别判定 A 相符。编译验收未执行(随 Phase 1 场地)。

### Phase 3 — 构建产物与打包链
- **目标**:产出可安装的 8.5.2 版 wheel/conda 包。
- **动作**:`run_build` 全链跑通(含 cupo runtime 变体选择、RPATH/SONAME 核对、`plugin_build.json` 元数据);`write_package_metadata.py` 钉扎 bump;`meta.yaml` 钉扎 bump;三处(§2.4)+ 文档表述同步。
- **产出**:候选 wheel + conda channel 包;`runtime-requirements.txt` 正确反映 `openmm[cuda13]>=8.5,<8.6`(具体上限见 D3)。
- **验收**:`ldd`/`readelf` 核对 RUNPATH 与 NEEDED 全解析;包安装进干净环境可 `import nv_openmm_plugin`。
- **状态(2026-11-20)**:钉扎机制已重构为探测驱动(§2.4 施工更新);`runtime-requirements.txt` 由 `write_package_metadata.py` 生成(`setup.py`/`pyproject.toml` 消费),全仓 grep 确认 plugins/openmm 之外无遗留 8.4 钉扎。run_build 全链、包产物与安装验收未执行(需 13.4 Docker 线)。

### Phase 4 — 运行时接入验证(GPU 节点执行;登录节点无 GPU)
- **目标**:证明插件在 8.5.2 上"活着"。
- **动作**:`import nv_openmm_plugin` 加载序验证(无 plugin load failures);CUDA 平台枚举;kernel 注册核查(重点:**bonded 三内核被本插件接管且无 `std::bad_cast`**——Kernels.h:39-54 记录过注册表冲突的爆炸模式);Context 创建、单步 step、checkpoint 存取。
- **产出**:冒烟清单及结果记录(每项:命令、节点、结果)。
- **验收**:冒烟全绿;`getPluginLoadFailures()` 为空;双加载防护仍有效。

### Phase 5 — 数值与性能回归(GPU 节点)
- **目标**:证明"迁移没改物理"。
- **动作**(全部走既有入口,不新造轮子):
  1. `./run_test openmm_unit`(141 项插件测试全量);
  2. 对拍套件:`test_plugin_vs_stock.py`、`test_integrator_vs_stock.py`(同 8.5.2 stock CUDA 平台对拍);
  3. golden:`test_golden_runs.py`/`golden_runs.py`(容差沿用现行阈值,不改判据);
  4. 物理端到端:`test_tyk2_alchemical.py`、`test_npt.py`、`test_nv_barostat.py`;
  5. 序列化/重启:`test_checkpoint_*`、`test_workflow_serialization.py`、`test_force_serialization.py`;
  6. 性能:`./run_test openmm_bench` 对比 8.4.0 基线(同机同卡),回退阈值建议 ≤3%(与 CMakeLists 里 ftz 决策的量级一致)。
- **产出**:回归报告(通过率、失败归类:迁移引入 vs 环境差异 vs 既有 flaky;性能对比表)。
- **验收**:unit 全过;对拍/golden 达标;性能在阈值内。任何失败禁止"调容差放行",必须归因。
- **输入文件事实(2026-11-20 实测)**:`_benchmark_systems.py` 从 OpenMM tag 下载的两个基准输入(apoa1.pdb、5dfr_solv-cube_equil.pdb)在 8.4.0 与 8.5.2 tag 下 SHA-256 逐字节相同(文件内钉扎哈希原样有效);届时把该文件 `OPENMM_RELEASE` 从 8.4.0 改 8.5.2 属表述性对齐,不影响下载校验。

### Phase 6 — 发布链收口
- **动作**:README/文档/commands.md 中版本表述更新;发布物(wheel/conda)以新版本号(见 D4)出;release notes 写明"requires OpenMM 8.5.x";CI 矩阵(若改)更新。
- **验收**:干净环境按 commands.md 三种安装方式(pip PyPI / wheel / conda)各装一次并跑 samples。
- **状态/更正(2026-11-20)**:①根与插件 README 首图 alt 文本描述的是 **8.4.0 上实测的基准数据**——该表述不随支持版本更新,只有重跑基准后才可改;支持性表述("requires OpenMM 8.5.x"等)待 Phase 5 全绿后统一收口。②版本号(D4):用户当日拍板**暂不动**,留到本阶段再定。

### Phase 7 — 8.6 增量 — **差异审计已预执行(2026-11-20,§3.2c),级别判定同为 A(纯重编译)**
- 已定案:8.5.2→8.6.1 的三处实质变更均不触及插件;`typemaps.i` 8.6 未再改;成员扫描零缺失;conda CUDA 13 变体在案。
- 施工内容(8.5.2 验收后执行):钉扎三处 bump 到 `>=8.6,<8.7`;双版本 configure;Phase 2-6 同流程重跑。
- 可选功能增强(独立决策,不属迁移):Nv barostat 跟进 8.6 的逐粒子独立缩放模式(`scaleMoleculesAsRigid=false` 语义)。

---

## 6. 验证矩阵

| 维度 | 取值 |
|---|---|
| OpenMM | 8.5.2(主);8.4.0(回归对照,证明单源码未破坏旧线);8.6.1(Phase 7,审计已定案 §3.2c) |
| CUDA 工具链 | 13.4(正式构建);12.9(仅 openmm_dev 联调,若 D2 允许) |
| 精度 | single / mixed(现有测试的既有组合) |
| 测试集 | unit 全量 + vs_stock 对拍 + golden + alchemical/NPT 端到端 + 序列化/重启 + bench |
| 执行位置 | **编译可在登录节点;一切 GPU 测试在计算节点提交**(登录节点无 GPU,用户 2026-11-20 明确) |
| 安装形态 | 源码树构建(`run_build`)、wheel 安装(`CUPO_TEST_INSTALLED=1`)、conda 安装(三选二以上) |

---

## 7. 风险登记册

| # | 风险 | 触发面 | 影响 | 缓解 | 回滚 |
|---|---|---|---|---|---|
| R1 | OpenMM 内部接口存在语义级变更(成员在但行为不同) | L2/L3 | bonded 接管与 integrator 行为偏移 | **已大幅降级**:§3.2b 头文件 diff 定案"零修改、纯新增";残余为行为级漂移,Phase 5 对拍兜底 | 源码回到 8.4 支路(单源码时关兼容宏) |
| R2 | swig runtime 版本错配 → `_nvplugin.so` 与 `_openmm.so` 类型体系断裂,`Context` 拒收插件 Integrator | L4 | Python 层全不可用 | **已实证排除**:真实 `NvPlugin.i` + swig 4.5.0 + 8.5.2 头实测通过(§3.3);构建机固定同代 swig(4.x) | 用匹配 swig 重出 `_nvplugin.so` |
| R3 | 钉扎漏改或过宽 → 求解器装错 OpenMM 系列,无版本 soname 静默错配(在案事故:>=8.4 静默解析到 8.6.0) | §2.4 三处 | 用户侧玄学崩溃 | 三处联动 checklist;加载时版本断言(§4.5) | 收紧钉扎重发包 |
| R4 | CUDA 12.9/13.4 双线混装 → cupo soname(`libcudart.so.13`/`libcufft.so.12`)在 12.9 环境缺失,或嵌入式表达式 IR 与 NVRTC 不匹配 | 环境 | 运行期 `CUFFT_MISSING_DEPENDENCY` 或 JIT 失败 | 双线严格分环境;fetch_cupo 按 CUDA 选变体;`plugin_build.json` 的 cuda_min 校验已有,保持 | 回到 13.4 单线 |
| R5 | 8.5 "下一代平台"改动波及 `olla` 路径假设(源码树构建时的 include 列表) | L5 | configure/编译失败 | **已降级**:头文件布局核实未变;Phase 1 双版本 configure 验证兜底 | 调整 include 列表,不影响逻辑 |
| R6 | golden/对拍出现小幅数值漂移(8.5 stock CUDA 平台 kernel 编译器/JIT 行为变化引入) | Phase 5 | 误判为迁移缺陷 | `--use_fast_math` 默认已核实未变(§3.2b);仍先跑 stock-vs-stock(8.5.2 内自对拍)建立环境基线,再做插件-vs-stock | 非插件原因则记录豁免,不改代码 |
| R7 | 环境操作越权(装包/升级触碰生产) | 全程 | 用户明令禁止 | 每个环境动作单列审批节点(§4.4);默认零环境改动 | — |
| R8 | 8.5.2 `CudaContext::popAsCurrent()` 新增栈平衡校验:弹出 context ≠ 预期即抛 `OpenMMException`(8.4 是静默弹出) | 插件 barostat/bonded 的 RAII `ContextSelector` 作用域内若发生跨 context 嵌套 push/pop | 运行期抛异常(比 8.4 的静默栈损坏更响、也更安全) | 插件从不直接调 push/pop,全部 RAII 配对;Phase 4 冒烟(barostat/GroupedBonded 路径)+ `test_npt`/`test_nv_barostat` 覆盖 | 若真踩到:检查选择器作用域内是否有第三方插件/linked context 嵌套,收窄作用域即可 |
| R9 | 8.6 `SerializationNode` 的 `getStringProperty`/`getProperties` 改为按值返回:取指针(`&node.getStringProperty(...)`)或把引用存出作用域的写法在 8.5.2 安全、8.6.1 悬垂 | 未来新增的序列化代码 | 隐蔽的悬垂引用/UB | 当前插件唯一绑定(`NonbondedForceProxy.cpp:149`)已核实为作用域内即用即弃、靠生命周期延长安全(§3.2c);代码规约:对这两个 getter 只按值接收(`const std::string s = ...`)或立即使用 | 出现悬垂症状(序列化读回乱码/偶发崩溃)时优先查新增的属性读取写法 |

---

## 8. 回滚与并存策略
- **发布物层面**:8.4 系 wheel/conda 包不下架;8.5.2 支持以新版本号发布(D4),用户可随时装回。
- **源码层面**:若 D1 采用单源码 + 兼容宏,8.4 支持随时可编译回退;若采用分支,迁移分支不动 `main` 直至 Phase 5 全绿。
- **环境层面**:`openmm_dev` 不做不可逆改动;任何需安装的动作先备份 `conda-meta` 清单(已有 `mamba_env_backup` 习惯,沿用)。

---

## 9. 决策点(需用户拍板后进入施工)

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| D1 | 源码组织 | (a) 单源码 + `NV_OPENMM_GE_85` 兼容宏,同时支持 8.4/8.5.2;(b) 分支/独立 tag | **(a)**,差异预期小,双版本 CI 可长期兜底 |
| D2 | `openmm_dev`(CUDA 12.9)内如何实际构建/联调 | (a) 仅做头文件/SWIG 级验证(已完成,见 §3.3),所有真构建走 13.4 Docker/计算节点;(b) 自建 cupo 12.9 在 openmm_dev 内完整构建——**受限**:cupo v0.3.5 release 只有 CUDA 13.4 tarball(§3.3),需 cupo 完整源码(本仓库树不含)且要装包(需审批) | **(a) 定案**:事实约束(cupo 无 12.9 变体)使 (b) 成本远超收益;`openmm_dev` 的角色收敛为"8.5.2 API 参照与头文件级验证环境" |
| D3 | 钉扎上限(8.5 线与 8.6 线) | (a) 各线封闭:`>=8.5,<8.6` 与 `>=8.6,<8.7`,分两次发布;(b) 一步放宽到 `<8.7` 同时覆盖 8.5/8.6 | **(a)** 延续现策略(避免重演">=8.4 静默解析到 8.6"教训);**8.6 已完成同标准审计(§3.2c),若 8.5.2 线 Phase 4/5 全绿,允许合并决策把 8.6 线提前**,由用户拍板 |
| D4 | 版本号 | (a) 0.3.6(仅兼容性);(b) 0.4.0(带钉扎/断言等行为变化) | **(b)**,钉扎变化对用户可见 |
| D5 | 加载时 OpenMM 版本断言(§4.5) | 加 / 不加 | **加**,成本极低,直接封死 R3 |

> **[拍板与落地状态 2026-11-20]** D1 按 (a) 落地(单源码 + `NV_OPENMM_GE_*` 宏,当前零调用点,纯保险);D2 按 (a) 定案(在案);D3 按 (a) 落地(系列封闭边界由探针自动推导,上限=下一 minor);D5 已落地(`__init__::_assert_openmm_series`,dev 版本号如 `8.5.2.dev-36a30cb` 可解析;元数据缺失时软放过,保持 soft-import 契约);D4 **未拍板**——版本号暂不动(用户当日拍板),留待 Phase 6。

---

## 10. 交付物清单
1. 本设计框架(本文档)。
2. Phase 0:《OpenMM 8.4.0→8.5.2 差异报告》(逐文件/逐符号)。
3. Phase 1:构建环境清单 + 双版本 configure 通过记录。
4. Phase 2:适配补丁集(按逻辑条目拆分,独立可回退)。
5. Phase 3:候选 wheel + conda 包 + 钉扎三处联动 diff。
6. Phase 4/5:冒烟记录、回归与性能报告。
7. Phase 6:发布说明与文档更新。

---

## 附:本框架依据的关键文件索引
- 构建链:`Dockerfile:82`、`plugins/openmm/run_build:79-89`、`plugins/openmm/CMakeLists.txt:89-141`
- 钉扎:`plugins/openmm/recipe/meta.yaml:61-72`、`plugins/openmm/write_package_metadata.py:31-44`
- 内核接管:`plugins/openmm/cpp/platforms/cuda/src/common/CudaKernelFactory.cpp:46-60`、`cpp/platforms/cuda/src/bonded/GroupedBonded.cu:306`
- SWIG:`plugins/openmm/cpp/swig/NvPlugin.i:25,93`
- 加载序与防双载:`plugins/openmm/python/nv_openmm_plugin/__init__.py:54-131`
- cupo 运行时取数:`fetch_cupo`(tarball 名含 CUDA 版本)、`include/cupo/version.h`(CUPO_ABI_VERSION=32)
- 8.5.2 事实来源:`openmm_dev` 环境内 OpenMM 8.5.2 安装(头文件/conda-meta/`version.py`)

---

## 附录 B:源码审计方法与产物(2026-11-20,可复现)

**全程零 git 操作**(下载 tarball 而非 clone)、零环境改动、零仓库改动(期间 SWIG 把生成的 `NvPlugin_wrap.cxx` 落进了仓库 `cpp/swig/`,当场删除恢复原状)。

1. **tarball 对比**:从 GitHub 官方 tag 下载 `openmm-8.4.0/8.5.0/8.5.1/8.5.2/8.6.0/8.6.1` 源码(各约 21 MB),解包到 `/tmp/omm-survey/<版本>/`(各一独立目录),`diff -rq` 全树比对 + 聚焦插件耦合面逐文件 `diff -u`;链式比对(8.4.0→8.5.2→8.6.1)用于把变更归因到具体小版本。仓库路径到安装路径的映射:`platforms/cuda/include/X.h` → `<env>/include/openmm/cuda/X.h`,`platforms/common/include/openmm/common/` → `<env>/include/openmm/common/`,`wrappers/python/src/swig_doxygen/swig_lib/python/typemaps.i` → `<env>/include/swig/typemaps.i`。
2. **release notes**:GitHub API `repos/openmm/openmm/releases` 原文(§3.1 即其整理)。
3. **SWIG 实测**(复现命令):
   ```bash
   <env>/bin/swig -python -c++ \
       -I/home/ruigengji/miniforge3/envs/openmm_dev/include \
       -I/home/ruigengji/cuPotential/plugins/openmm/cpp/openmmapi/include \
       /home/ruigengji/cuPotential/plugins/openmm/cpp/swig/NvPlugin.i -outdir /tmp/...
   ```
   结果 exit=0,仅 Warning 490(Py_StripOpenMMUnits fragment 改为 %header 注入,良性)/453/314(OpenMM 自有接口警告,8.4 构建同样存在)。fragment 机制单独最小样例验证:函数确实经 `%header` 注入生成的 `.cxx`。
4. **conda 变体**:anaconda.org API `package/conda-forge/openmm`,列出 8.5.2 linux-64 全部构建的 depends;本地 `openmm_dev/conda-meta/openmm-8.5.2-*.json` 对照。
5. **cupo 变体**:GitHub API `repos/NVIDIA/cuPotential/releases` v0.3.5 资产清单(SHA256SUMS 在案)。
6. 产物位置(会话内临时,供追溯):`/tmp/omm-survey/`(tarball、四棵源码树、关键 diff 文件、releases/conda JSON)。如需长期留存差异报告,可在 Phase 1 后把结论导出为正式文档入 `docs/`(届时另行征求用户同意)。
