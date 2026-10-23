# ReflectoFit

[![Tests](https://github.com/NaNIsaac/ReflectoFit/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/ReflectoFit/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

빠른 X선 반사율(XRR) 피팅 엔진과 명령줄 도구: 층 모델, 기기 모델, 최소제곱, differential evolution과 MCMC, 모델 선택, 피팅 보고.

박막의 specular XRR 곡선을 측정했고 각 층의 두께, 거칠기, 밀도를 얻고 싶은 사람을 위한 도구다. 순방향 모델은 float64 Parratt 재귀(numba 컴파일)에 Névot-Croce 거칠기, 가우시안 분해능, 빔 footprint, Cu Kα₂, 2θ offset을 더한 것이고 refnx와 1e-10 안에서 일치한다. torch backend는 같은 곡선을 gradient와 함께 준다. 짝이 되는 시뮬레이터 ReflectoSim은 이 순방향 모델 위에서 기계학습용 합성 XRR 데이터를 만든다.

상태: 초기 개발 단계. 설계는 [`docs/DESIGN.md`](docs/DESIGN.md)에 있고, 모델 선택은 뒤에 들어온다.

## 설치

아직 PyPI에 없다. clone한 뒤 [uv](https://docs.astral.sh/uv/)로:

```sh
git clone https://github.com/NaNIsaac/ReflectoFit
cd ReflectoFit
uv sync
```

## 첫 실행

Si 위 SiO₂ 위의 240 Å 막을 Poisson 계수로 측정한 곡선을 만들고, 틀린 시작값에서 피팅한다:

<!-- test -->
```sh
uv run reflectofit calc --model examples/film.toml --counts 1e8 -o film.dat
uv run reflectofit fit film.dat --model examples/film_start.toml --loss poisson --counts-scale 1e8 -o fit
```

피팅은 자유 파라미터마다 값과 표준오차를 출력하고 `fit/result.json`, `fit/model.toml`(피팅된 모델, 다음 시작값으로 재사용 가능), `fit/fitted.dat`를 쓴다.

## 모델

모델은 TOML 파일이다: 기기, 위에서부터의 층, 기판, 그리고 범위를 준 자유 파라미터.

```toml
[instrument]
wavelength = 1.5406    # Å
i0 = 1.0
background = 3e-8
sigma_q = 5e-4         # q에서의 가우시안 분해능 σ [Å⁻¹]; dq_over_q는 q에 비례하는 항을 더한다
sample_length = 10.0   # mm, footprint에는 beam_sigma(mm)와 함께
beam_sigma = 0.04

[[layer]]
name = "film"
sld = [45.0, 1.5]      # 1e-6 Å⁻², [실수, 허수]; 허수 > 0이 흡수
thickness = 240.0      # Å
roughness = 6.0        # Å, 이 층 위쪽 계면의 거칠기

[[layer]]
name = "oxide"
material = "SiO2"      # 또는 조성과 밀도로 SLD를 준다
density = 2.20         # g/cm³
thickness = 12.0
roughness = 3.0

[substrate]
material = "Si"
density = 2.33
roughness = 3.0

[fit]
"film.thickness" = [150.0, 350.0]
"film.sld" = [20.0, 80.0]
i0 = [0.5, 2.0]
log10_background = [-10.0, -5.0]
```

자유 파라미터: 이름 있는 각 층과 `substrate`의 `<layer>.thickness`, `.roughness`, `.sld`, `.isld`; 기기 항목 `i0`, `background`, `log10_background`, `tth_offset`, `sigma_q`, `dq_over_q`, `kalpha2_ratio`, `sample_length`, `beam_sigma`.

## 명령줄

| 명령 | 하는 일 |
|---|---|
| `reflectofit calc --model M -o out.dat` | 격자(`--grid tth:0.1:6:300`) 위의 모델 곡선, 원하면 Poisson 계수까지 |
| `reflectofit fit data.dat --model M` | 최소제곱(`--method lsq`, `--trust 0.2`는 각 시작값을 ±20 %로 제한) 또는 differential evolution(`--method de`); 손실 `log`, `raw`, `poisson`, `chi2` |
| `reflectofit score data.dat --model M` | 데이터가 허용하는 모든 손실, likelihood 손실에는 AIC와 BIC |
| `reflectofit stitch 1.dat 2.dat -o all.dat` | 서로 다른 감쇠기로 측정한 scan 잇기(`--method boundary` 또는 `median`) |

데이터 파일은 2θ(도, 또는 `--x q`면 q, Å⁻¹), 세기, 선택적 오차 열로 된 텍스트다. `--counts-scale S`를 주면 세기를 계수로 읽고 R = 1에서 S 계수로 본다. Poisson 손실에는 이것이 필요하다.

## Python

```python
import numpy as np
from reflectofit import Instrument, Layer, Stack, forward, materials, tth_to_q

stack = Stack(
    [Layer(45 + 1.5j, 240, 6, "film"), Layer(materials.sld("SiO2", 2.20), 12, 3, "oxide")],
    Layer(materials.sld("Si", 2.33), roughness=3),
)
q = tth_to_q(np.linspace(0.1, 6, 300))
r = forward(q, stack, Instrument(sigma_q=5e-4, background=3e-8))
r_torch = forward(q, stack, backend="torch")  # 미분 가능, torch extra 필요
```

## 개발

```sh
uv run pytest
```

## 라이선스

MIT
