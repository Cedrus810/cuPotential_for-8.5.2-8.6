[English](README.md) | **日本語** | [中文](README.zh.md)

# cuPotential OpenMM プラグイン — OpenMM 8.6 対応ラボ版（`0.3.5+openmm86`）

このリポジトリは [NVIDIA cuPotential](https://github.com/NVIDIA/cuPotential) **v0.3.5** をフォークしたラボ版です。OpenMM プラグインを **OpenMM 8.6** 向けにビルドし直し、エネルギー最小化に修正を 1 点加えています。**NVIDIA の公式リリースではありません。** NVIDIA の 0.3.5（PyPI の `nv-openmm-plugin` を含む）は OpenMM 8.4 向けです。NVIDIA のオリジナル README は [README.NVIDIA.md](README.NVIDIA.md) に残してあります。API・理論・サンプルは変更していないので、[cuPotential ドキュメント](https://docs.nvidia.com/cuda/cupotential)を参照してください。

## NVIDIA 0.3.5 との違い

- **OpenMM 8.6 対応**
  - プラグインのソースは変更なしで OpenMM 8.6 に対してビルドできます。
  - ビルド時に CMake が OpenMM のバージョンを検出し、その結果から次の 3 つを決めます。
    - wheel の依存（`openmm[cuda13]>=8.6,<8.7`）
    - conda レシピのバージョン固定
    - import 時のバージョンチェック（8.6 系列以外の OpenMM では読み込みを拒否します）
- **OpenMM 8.5 以降のエネルギー最小化の修正**
  - 8.5 から OpenMM は最小化を GPU で行います。拘束のある System では、拘束を表すばねの力がいずれ GPU の固定小数点力バッファをあふれさせます。すると OpenMM は System 全体を CPU 上に作り直しますが、そこにはプラグインの力のカーネルがないため、`Specified a Platform for a Context which does not support all required kernels` で失敗します。
  - 本ビルドでは、プラグインの力（`nv.NonbondedForce`、`nv.MonteCarloBarostat`）を含む System を OpenMM 8.4 と同じ方法で最小化します。L-BFGS をホスト上で実行し、ばねの力もホスト上の倍精度で加えます。8.4 のアルゴリズムであり、8.5 以降の GPU 最小化は使わないため、それより遅くなります。
  - プラグインの力を含まない System は、従来どおり OpenMM 自身の GPU 最小化を使います。
  - System 自体の力の成分が約 2e9 kJ/mol/nm に達した場合（原子の重なりなど）は、CPU に切り替えず、原因を説明するエラーを出します。
- **バージョン `0.3.5+openmm86`**：PEP 440 のローカルラベルを付けたので、NVIDIA の 0.3.5 と区別できます。

API と使い方は NVIDIA 0.3.5 と同じです。

## 動作要件

| | |
|---|---|
| OS | Linux x86_64 |
| Python | **3.14 のみ**（wheel にビルド済みの SWIG 拡張が入っており、タグが `cp314` のため） |
| OpenMM | CUDA 13 プラットフォーム付きの 8.6.x（pip が PyPI から入れます） |
| CUDA ライブラリ | CUDA runtime・NVRTC・nvJitLink `>=13.4,<13.5`、cuFFT `>=12.4,<12.5`（pip が入れます。CUDA ツールキットは不要） |
| NVIDIA ドライバ | CUDA 13.4 に対応しているもの（615.71 で確認済み。580.x は未確認） |
| GPU | sm_75〜sm_120（Turing〜Blackwell）。確認は RTX 5090 のみ |

NVIDIA の `nv-openmm-plugin` とはパッケージ名が同じなので、同じ環境に入れないでください。

## インストール

このリポジトリの **Releases**（タグ `v0.3.5+openmm86`）から `nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl` をダウンロードし、次を実行します。

```bash
python3.14 -m venv omm86 && source omm86/bin/activate     # python=3.14 の conda 環境でも可
pip install --upgrade pip
pip install nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
python -c "import openmm, nv_openmm_plugin; print(openmm.version.full_version, [f for f in openmm.Platform.getPluginLoadFailures() if 'nv_openmm_plugin' in f])"
```

最後のコマンドで 8.6.x のバージョンと `[]` が表示されれば成功です。`LD_LIBRARY_PATH` の設定は不要です。

既存の conda-forge 版 OpenMM 8.6 環境に `pip install --no-deps` で入れる方法は未確認です。

## 確認済みの環境

確認した環境：RTX 5090、ドライバ 615.71、OpenMM 8.6.1、CUDA 13.4.92、Python 3.14.7。

| 実行内容 | 結果 |
|---|---|
| wheel を新しい venv にインストール | import 成功。CUDA ライブラリはすべて venv 内のものが読み込まれる |
| wheel から `samples/solvated_systems` | 成功 |
| wheel からインストール済みテスト `test_unit_conversion` / `test_npt` | 成功 |
| ソースツリーでプラグインのテスト一式 | すべて成功（`cupy` が必要な `test_leak_detection` は未実行） |
| ソースツリーでベンチマーク（GPU クロック未固定） | 同じ GPU 上の標準 OpenMM 8.6.1 に対し、1 ステップあたり 2.3〜3.8 倍 |

リリースノート（英語）：[plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md](plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md)

## ソースからのビルド（メンテナ向け）

次のものが必要です。

- `openmm=8.6`、`python=3.14`、`swig`、`cmake>=3.24`、`cuda-version=13.4` を入れた conda 環境
- CUDA 13.4 ツールキット（`nvcc`）
- gcc

Docker は不要です。

```bash
E=<上記の conda 環境>
export PATH=<cuda-13.4>/bin:$E/bin:$PATH OPENMM=$E
CUPO_REPOSITORY=NVIDIA/cuPotential python3 fetch_cupo      # NVIDIA v0.3.5 の cupo ランタイム（CUDA 13.4）
plugins/openmm/run_build
GITHUB_RELEASE_TAG=v0.3.5 python -m pip wheel --no-build-isolation --no-deps -w dist plugins/openmm
```

ルートの `VERSION`／`RELEASE_TAG` は `0.3.5`／`v0.3.5` のままにしてください。`fetch_cupo` がこれを使って cupo ランタイムを取得します。パッケージのバージョンは `plugins/openmm/VERSION` にあります。

## ライセンス

ライセンスは NVIDIA cuPotential 0.3.5 と同じです。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。新しい最小化コードは OpenMM 8.4.0 の `LocalEnergyMinimizer.cpp`（MIT）に基づき、OpenMM 同梱の libLBFGS ヘッダ（MIT）を含みます。どちらも `NOTICE` に記載しています。
