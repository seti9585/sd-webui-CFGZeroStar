# sd-webui-CFGZeroStar

**EN** | [日本語](#日本語)

Post-CFG guidance extension for Stable Diffusion WebUI (Forge-based).
Rescales the unconditional branch by the least-squares optimal factor α (optimized scale), and optionally zeroes the model prediction for the first few steps (zero-init).
Operates in x0 (denoised) space — prediction-space agnostic, so it works on both SDXL (UNet/epsilon) and flow-matching (Anima) models.

Paper: [arXiv:2503.18886](https://arxiv.org/abs/2503.18886)
Original: [WeichenFan/CFG-Zero-star](https://github.com/WeichenFan/CFG-Zero-star) / ComfyUI built-in `CFGZeroStar` node

> Recommended CFG: 7–10. Stacking multiple CFG-axis extensions at very high CFG (≈20+) can cause error accumulation.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-CFGZeroStar
```

---

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| Enable CFG-Zero\* | off | Enable the optimized-scale correction |
| Enable zero-init | off | Zero the model prediction for the first N steps |
| Zero-init steps | 0 | Steps to zero. `0` = auto (~4% of total steps) |

---

## Algorithm

### Optimized scale

```
v_cond   = x_t − cond_denoised
v_uncond = x_t − uncond_denoised
α        = <v_cond, v_uncond> / (‖v_uncond‖² + 0.00000001)   ← per-sample scalar
output   = out + uncond_denoised × (α − 1)
               + cond_scale × uncond_denoised × (1 − α)
```

`out` is the standard CFG result. The additive form (`out + delta`) preserves any earlier post-CFG corrections (FreSca, MaHiRo, TCFG).

α is computed in x0 space — the sampler wrapper converts the raw model output (epsilon / velocity / flow-matching) to an x0 estimate before the post-CFG hook fires, so α does not depend on the model's prediction parameterization.

Measured behaviour (reForge / Illustrious / CFG 7 / RK Sampler): α ≈ 0.999 at the first step, drifting to ≈ 0.964 at the low-σ tail. The correction is largest at the late (low-σ) steps that govern tone and contrast, leaving composition intact.

### Zero-init

```
if σ > σ_schedule[N]:
    denoised = 0   →   x_next = x × (σ_next / σ)
```

Zeroing the model prediction causes the latent to rescale to track σ, skipping the early unreliable estimate. "First N steps" is located from the sigma schedule when the backend exposes it; otherwise an approximate log-σ fraction fallback is used automatically.

> **Note:** zero-init is validated on both SDXL and Anima through Forge's σ-based ODE sampler interface.
> It re-rolls early composition (larger N = more variation) but does **not** improve image quality on either architecture — contrast and saturation decrease at high N. Keep it at the auto default (≈ 1 step at 32 steps) and treat it as a composition-variation knob, not an enhancement.
> The paper's quality benefit is specific to native flow-matching pipelines and does not transfer to Forge's unified σ sampler.

---

## Behaviour in a multi-extension post-CFG chain

Forge-based backends allow several extensions to register a post-CFG hook.

The additive output form makes this hook tolerant of being placed anywhere in the queue: earlier corrections are carried forward rather than overwritten. Tolerance is not determinism, however. Without an explicit position the hook is appended in extension load order, which follows alphabetical order, so the same settings can yield different results depending only on which other extensions happen to be installed.

This extension therefore exposes a `_sd_webui_priority` attribute and inserts its hook at priority 15.0 inside the post-CFG queue. Other extensions in the same suite that expose the attribute participate in the same ordering.

Note that this is separate from `sorting_priority`, which controls only the display order of the UI accordion and has no effect on callback execution order.

The registration log reports the resolved priority. The authoritative record of actual call order is the sampling-time chain dump emitted by [sd-webui-FreSca](https://github.com/seti9585/sd-webui-FreSca), which lists every registered post-CFG hook in the order it runs.

---

## Debug output

Set the environment variable `SD_WEBUI_SETI_DEBUG` before launching:

| Value | Effect |
| --- | --- |
| 0 (or unset) | No debug output |
| 1 | Registration logging, plus per-step reporting of the resolved path and the mean, minimum, and maximum of α |

The per-step α values are useful for confirming that the correction behaves as expected across the sigma schedule, and for distinguishing a genuinely inactive correction (α ≈ 1 throughout) from one that never registered.

Output goes to both the module logger and stderr, because some backends suppress module-level logger output.

---

## Tested environments

- reForge (Python 3.10) — SDXL-family models
- Forge Neo (Python 3.12) — SDXL-family and Anima (flow-matching DiT), txt2img + Hires.fix

Not compatible with A1111 (`set_model_sampler_post_cfg_function` is Forge-backend only).

---

# 日本語

**[English](#sd-webui-cfgzerostar)** | 日本語

Forge 系 WebUI 向け Post-CFG ガイダンス拡張機能。
無条件分岐を最小二乗最適スケール α（optimized scale）でリスケールし、オプションで最初の数ステップのモデル予測をゼロ化（zero-init）します。
x0（denoised）空間で動作するため、予測パラメータ化（epsilon / velocity / flow-matching）に依存せず、SDXL（UNet）と flow-matching（Anima）の両方で機能します。

論文: [arXiv:2503.18886](https://arxiv.org/abs/2503.18886)
原実装: [WeichenFan/CFG-Zero-star](https://github.com/WeichenFan/CFG-Zero-star) / ComfyUI ビルトイン `CFGZeroStar` ノード

> 推奨 CFG: 7〜10。CFG 軸の拡張機能を複数重ねて CFG を高くしすぎると（≈20+）、誤差が累積します。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-CFGZeroStar
```

---

## パラメータ

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| Enable CFG-Zero\* | off | optimized-scale 補正を有効にする |
| Enable zero-init | off | 最初の N ステップのモデル予測をゼロ化する |
| Zero-init steps | 0 | ゼロ化するステップ数。`0` = 自動（総ステップ数の約 4%） |

---

## アルゴリズム

### Optimized scale

```
v_cond   = x_t − cond_denoised
v_uncond = x_t − uncond_denoised
α        = <v_cond, v_uncond> / (‖v_uncond‖² + 0.00000001)   ← サンプルごとのスカラー
output   = out + uncond_denoised × (α − 1)
               + cond_scale × uncond_denoised × (1 − α)
```

`out` は通常の CFG 結果です。加算形（`out + delta`）により、先行する Post-CFG フック（FreSca・MaHiRo・TCFG）の補正を上書きせず積み重ねます。

α は x0 空間で計算されます。Post-CFG フックが呼ばれる時点で、サンプラーがモデルの生出力（epsilon / velocity / flow-matching）を x0 推定値に変換済みのため、α はモデルの予測パラメータ化に依存しません。

実測挙動（reForge / Illustrious / CFG 7 / RK Sampler）: α は第0ステップで ≈ 0.999、低 σ 末尾で ≈ 0.964 まで下降します。補正は後半（低 σ）ステップ ── トーンとコントラストを決める領域 ── に集中し、構図はそのまま維持されます。

### Zero-init

```
if σ > σ_schedule[N]:
    denoised = 0   →   x_next = x × (σ_next / σ)
```

モデル予測をゼロ化することで、latent が σ に追従して縮み、初期の不安定な推定をスキップします。「最初の N ステップ」の判定はバックエンドが sigma スケジュールを公開している場合は厳密版、非公開の場合は自動的に log-σ 分率フォールバックを使用します。

> **注意:** zero-init は SDXL・Anima の両アーキテクチャについて、Forge の σ ベース ODE サンプラーインターフェース上で検証済みです。
> N を大きくするほど初期構図が変化しますが、いずれのアーキテクチャでも画質は向上しません ── 高 N でコントラスト・彩度が低下します。自動（32ステップで ≈ 1ステップ）のまま使用し、品質機能ではなく構図バリエーションのノブとして扱ってください。
> 論文の品質上の利点はネイティブ flow-matching パイプライン固有のもので、Forge の統一 σ サンプラー上では再現されません。

---

## 複数拡張機能が同時に動作する場合の挙動

Forge 系バックエンドでは複数の拡張機能が post-CFG フックを登録できます。

本拡張機能の出力は加算形のため、キュー内のどの位置に置かれても先行する補正を上書きせず引き継ぎます。ただし「順序に強い」ことと「順序が決まっている」ことは別です。位置を明示しない場合、フックは拡張機能のロード順（実質アルファベット順）で末尾に追加されるため、同じ設定でも他にどの拡張機能がインストールされているかによって結果が変わり得ます。

そこで本拡張機能は `_sd_webui_priority` 属性を持ち、post-CFG キュー内の優先度 15.0 の位置にフックを挿入します。同属性を持つ同シリーズの他の拡張機能も、同じ順序付けに参加します。

これは `sorting_priority` とは別物である点に注意してください。`sorting_priority` は UI アコーディオンの表示順のみを制御し、コールバックの実行順序には影響しません。

登録時のログには解決された優先度が出力されます。実際の呼び出し順序を示す正式な記録は、[sd-webui-FreSca](https://github.com/seti9585/sd-webui-FreSca) がサンプリング時に出力するチェーンダンプです。登録済みの全 post-CFG フックが実行順に列挙されます。

---

## デバッグ出力

起動前に環境変数 `SD_WEBUI_SETI_DEBUG` を設定します。

| 値 | 動作 |
| --- | --- |
| 0（または未設定） | デバッグ出力なし |
| 1 | 登録時のログ、および各ステップでの解決済みパスと α の平均・最小・最大の出力 |

ステップごとの α 値は、sigma スケジュール全体にわたって補正が期待通りに振る舞っているかの確認に有用です。また、補正が実質的に働いていない状態（全域で α ≈ 1）と、そもそも登録されていない状態とを区別できます。

一部のバックエンドがモジュールレベルのロガー出力を抑制するため、出力はモジュールロガーと stderr の両方に送られます。

---

## 動作確認環境

- reForge（Python 3.10）— SDXL 系モデル
- Forge Neo（Python 3.12）— SDXL 系および Anima（flow-matching DiT）、txt2img + Hires.fix

A1111 非対応（`set_model_sampler_post_cfg_function` は Forge バックエンド専用）。

---

## ライセンス

MIT License — Copyright (c) 2026 seti9585
Original: [WeichenFan/CFG-Zero-star](https://github.com/WeichenFan/CFG-Zero-star) (Apache-2.0) / ComfyUI built-in `CFGZeroStar` node
Based on: [arXiv:2503.18886](https://arxiv.org/abs/2503.18886)
