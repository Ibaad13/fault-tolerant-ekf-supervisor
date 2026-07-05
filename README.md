# 🛰️ UAV Fault-Tolerant Localization — Dual-EKF Sensor Fusion with a Confidence-Weighted Adaptive Supervisor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/Use-Academic%20%2F%20Research-lightgrey)
![Filters](https://img.shields.io/badge/EKFs-2%20parallel%2C%2015--state-orange)
![Fusion](https://img.shields.io/badge/Fusion-Adaptive%20Confidence--Weighted-purple)

A 15-state tightly-coupled Extended Kalman Filter (EKF) framework for 3D quadcopter
localization under **contested RF / GPS environments** (spoofing, jamming, NLOS
multipath). Two independent EKFs — one fusing IMU + multi-anchor RSSI ranging, the
other fusing IMU + GPS + magnetometer + barometer — run in parallel under an
**adaptive Supervisor** that continuously scores each filter's trustworthiness from
its innovation statistics and gating outcomes, and fuses the two into a single,
smoothly re-weighted navigation solution — no hard switches, no chattering, and a
guaranteed minimum voice for RSSI even while GPS looks perfectly healthy.

This repository accompanies a study on resilient multi-sensor navigation for UAVs
operating in adversarial or degraded-sensor conditions.


> [!WARNING]
> I haven't uploaded complete Python code as I am currently preparing the paper for a conference; only results are shared here.
> 📽️ **See it in action** — jump straight to the [animated trajectory results](#-results) below.




---

## Table of Contents

1. [System Overview](#system-overview)
2. [Shared 15-State Model](#shared-15-state-model)
3. [Algorithm 1 — IMU + Multi-Anchor RSSI EKF](#algorithm-1--imu--multi-anchor-rssi-ekf)
4. [Algorithm 2 — IMU + GPS + Magnetometer + Barometer EKF](#algorithm-2--imu--gps--magnetometer--barometer-ekf)
5. [The Supervisor — Context-Aware Adaptive Confidence-Weighted Fusion](#the-supervisor--context-aware-adaptive-confidence-weighted-fusion)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [📊 Results](#-results)
9. [🎬 Simulation Animations](#-simulation-animations)
10. [Citation / Acknowledgements](#citation--acknowledgements)

---

## System Overview

Both filters share the **same 15-state process model** and the **same sequential,
scalar-update EKF machinery** (predict once with IMU, then apply each aiding
measurement one scalar at a time with an individual Mahalanobis gate). They differ
only in their *aiding* sensor suite:

| | Algorithm 1 | Algorithm 2 |
|---|---|---|
| **Aiding sensors** | Multi-anchor RSSI ranging (5 anchors) | GPS position + velocity (6 channels) |
| **Shared aiding** | Magnetometer (heading) + Barometer (altitude), both @ 20 Hz | Magnetometer (heading) + Barometer (altitude), both @ 20 Hz |
| **Update rate (primary aiding)** | 10 Hz | 10 Hz |
| **IMU rate** | 100 Hz | 100 Hz |
| **Attack model** | Sudden RF spoofing step on one anchor, NLOS attenuation step on another | GPS position spoofing (step + slow walk-off), GPS velocity jamming (variance inflation) |
| **Gating** | Per-anchor Mahalanobis gate | Per-channel Mahalanobis gate |

Because both filters expose an *identical* state vector, `predict()` /
`_scalar_update()` interface, and gating result format, a **Supervisor** can treat
them as interchangeable black boxes: run both off the same IMU sample, watch their
gate accept/reject streams, and fuse or hand off between them without needing to
understand either filter's internals.

---

## Shared 15-State Model

Both EKFs estimate the same state vector, expressed in the **NED (North-East-Down)**
navigation frame for position/velocity/attitude and the **body frame** for IMU biases:

```
x = [ p_N, p_E, p_D,            position          (m)
      v_N, v_E, v_D,            velocity          (m/s)
      φ, θ, ψ,                  Euler attitude     (rad)   (roll, pitch, yaw)
      b_ax, b_ay, b_az,         accelerometer bias (m/s²)
      b_gx, b_gy, b_gz ]        gyroscope bias      (rad/s)
```

`N_STATES = 15`.

### Continuous-time process model

Given corrected specific force `a_corr = a_raw - b_a` and corrected body rate
`ω_corr = ω_raw - b_g`, the nonlinear continuous dynamics are:

```
ṗ      = v
v̇      = R_bn(φ,θ,ψ) · a_corr + [0, 0, g]ᵀ
[φ̇,θ̇,ψ̇]ᵀ = T(φ,θ) · ω_corr
ḃ_a    = 0        (random-walk bias, modeled via process noise Q, not drift here)
ḃ_g    = 0
```

where `R_bn` is the body-to-NED rotation matrix (Z-Y-X Euler convention):

```
R_bn(φ,θ,ψ) =
⎡ cψcθ   cψsθsφ − sψcφ   cψsθcφ + sψsφ ⎤
⎢ sψcθ   sψsθsφ + cψcφ   sψsθcφ − cψsφ ⎥
⎣ −sθ    cθsφ             cθcφ          ⎦
```

and `T(φ,θ)` is the Euler-rate kinematic (body-rate → Euler-rate) matrix:

```
T(φ,θ) =
⎡ 1   sφ·tanθ   cφ·tanθ ⎤
⎢ 0   cφ        −sφ      ⎥
⎣ 0   sφ/cθ     cφ/cθ    ⎦
```

(`cosθ` is clipped away from zero to avoid the pitch-90° singularity.)

### Discrete-time linearization (numeric Jacobian)

Rather than deriving the 15×15 analytical Jacobian by hand, both filters compute
`A = ∂f/∂x` via **central-difference numeric differentiation** (`eps = 1e-6`) of the
continuous model above, then form the discrete transition matrix with a first-order
(Euler) expansion:

```
F_k = I₁₅ + A · dt
P_k⁻ = F_k · P_{k−1} · F_kᵀ + Q · dt
```

`Q = diag([σ_p², σ_p², σ_p²,  σ_v², σ_v², σ_v²,  σ_att², σ_att², σ_att²,  σ_ba², σ_ba², σ_ba²,  σ_bg², σ_bg², σ_bg²])`

This keeps the linearization exact to numerical precision without hand-deriving
partials for every trigonometric cross-term in `R_bn` and `T`.

### Sequential scalar measurement update

Every aiding measurement (anchor range, GPS channel, heading, altitude) is applied
**one scalar at a time**, in sequence, rather than as a single stacked vector update.
For a scalar measurement `z` with model `h(x)` and Jacobian row `H`:

```
y   = z − h(x)                          (innovation)
S   = H·P·Hᵀ + R                        (innovation covariance, scalar)
ε   = y² / S                             (Mahalanobis / NIS statistic, χ²₁-distributed)
K   = P·Hᵀ / S                           (Kalman gain, column vector)
```

**Gating**: the update is only *committed* if `ε ≤ γ_thresh` (default `γ_thresh = 5.0`,
a chi-square gate at 1 DOF). If the gate rejects the sample:

```
x ← x + K·y
P ← (I₁₅ − K·H)·P
```

If rejected, `x` and `P` are left untouched — the measurement is discarded outright
rather than down-weighted, which is what allows spoofed/jammed samples to be
completely isolated from corrupting the state estimate.

---

## Algorithm 1 — IMU + Multi-Anchor RSSI EKF

**File:** [`filters/ekf_rssi.py`](filters/ekf_rssi.py)

### Aiding measurements

**1. Anchor ranging (RSSI → distance, 10 Hz, gated per anchor)**

Five fixed ground anchors at known NED positions (`config.ANCHORS`). Each anchor's
raw RSSI sample follows a log-distance path-loss model:

```
p_rx = P_TX − 10·n·log10(d_true / d₀) + w,      w ~ N(0, σ_RSSI²)
```

The raw RSSI is smoothed with an **Exponential Moving Average (EMA)**:

```
p̂_rx[k] = α·p_rx[k] + (1−α)·p̂_rx[k−1],      α = 0.5
```

then converted back to a metric distance + its effective measurement variance:

```
d_meas    = d₀ · 10^((P_TX − p̂_rx)/(10n))
σ_eff     = σ_RSSI · √(α / (2−α))                (EMA noise-reduction factor)
R_i       = ( (ln10 / (10n)) · d_meas )² · σ_eff²
```

The anchor measurement model (nonlinear range to a known 3D anchor position `r`):

```
h_i(x) = ‖ [p_N,p_E,p_D] − r_i ‖₂
H_i = [ (p_N−r_N)/h, (p_E−r_E)/h, (p_D−r_D)/h, 0, ..., 0 ]     (1×15, nonzero only in position block)
```

**2. Magnetometer heading (20 Hz, shared with Algorithm 2)**

```
y = atan2( sin(ψ_meas − ψ_est), cos(ψ_meas − ψ_est) )     (wrapped angular innovation)
H = [0,...,0, 1(at ψ index), 0,...,0]
```

**3. Barometer altitude (20 Hz, shared with Algorithm 2)**

```
y = p_D,meas − p_D,est
H = [0, 0, 1, 0, ..., 0]
```

### Contested-RF attack model

Starting at `attack_start_t` (default: halfway through the run):

- **Anchor 1 (`SPOOF_ANCHOR_IDX`)** — sudden **+25 m** malicious range spoof step.
- **Anchor 3 (`NLOS_ANCHOR_IDX`)** — sudden **+20 m** NLOS blockage attenuation step.

Both should trip the χ² gate (`ε > γ_thresh`) once the EMA lag catches up to the step,
causing those anchors to be rejected while the 3 clean anchors continue updating
normally.

---

## Algorithm 2 — IMU + GPS + Magnetometer + Barometer EKF

**File:** [`filters/ekf_gps.py`](filters/ekf_gps.py)

Structured as a drop-in "twin" of Algorithm 1 — identical state vector, identical
`predict()`/`_scalar_update()` machinery — so a Supervisor (or a Monte Carlo study)
can consume both filters' logs interchangeably.

### Aiding measurements

**1. GPS position/velocity (10 Hz, gated per channel)**

Six independent scalar channels `[p_N, p_E, p_D, v_N, v_E, v_D]`, each measured
directly and gated individually (mirroring Algorithm 1's per-anchor gating):

```
h_ch(x) = x[ch]                     (direct state observation, ch ∈ {0..5})
H_ch    = e_ch                       (unit row vector, 1 at state index ch)
R_pos   = σ_GPS,pos²  = 1.5² m²
R_vel   = σ_GPS,vel²  = 0.1² (m/s)²
```

**2. Magnetometer heading + Barometer altitude** — identical equations to Algorithm 1
(shared aiding sensors, same noise models, same 20 Hz rate).

### Contested-GPS attack model

Starting at `attack_start_t`:

- **North position channel (`p_N`)** — spoofing: a **+25 m step** plus a **0.6 m/s
  continuous walk-off**, i.e. `Δp_N(t) = 25 + 0.6·(t − t_attack)`.
- **East velocity channel (`v_E`)** — jamming: measurement variance inflated **400×**,
  which should cause the χ² gate to reject most (but not necessarily all) samples on
  that channel due to the ballooned `R`.

---

## The Supervisor — Context-Aware Adaptive Confidence-Weighted Fusion

**File:** [`filters/supervisor.py`](filters/supervisor.py)

The Supervisor owns one instance of each EKF, steps both with the **same IMU
sample** every tick (so their internal state trajectories only diverge due to their
different aiding sensors), and fuses their outputs into one navigation solution.

> **Old design (hard switching):** GPS/RSSI healthy flags fed a truth table that
> jumped between `GPS`, `RSSI`, `BLEND`, or `DEAD_RECKONING` — a filter judged
> "healthy" got 100% say, a filter judged "degraded" got 0%. Brittle, and RSSI had
> zero influence the instant GPS looked fine.
>
> **Current design (this repo):** no hard switch at all. Each filter carries a
> **continuous trust score** that smoothly re-weights a per-state inverse-covariance
> blend every tick — so a filter that's "70% trustworthy" contributes roughly 70% of
> the say it would at full health, not 0% or 100%.

### 1. Continuous health score (per filter, every tick)

Each filter's *aggregate* gate outcomes are tracked in a rolling window
(`WINDOW_LEN = 20` recent scalar updates) and combined into a health **score** in
`[0, 1]`, not just a boolean flag:

```
score_rejection = clip(1 − rejection_rate / REJECTION_RATE_DEGRADED,      0, 1)
score_nis       = clip(1 − mean_mahalanobis / MEAN_EPS_DEGRADED,          0, 1)
score_quality   = clip(good_anchors_or_channels_this_tick / total,        0, 1)

health_score  = 0.45·score_rejection + 0.20·score_nis + 0.35·score_quality
health_score ← EMA-smoothed over time (HEALTH_SCORE_EMA_ALPHA)
```

`score_quality` is the raw signal-quality proxy called for in the design brief —
fraction of RSSI anchors currently passing the gate (≈ "# good anchors"), or
fraction of GPS channels passing the gate (a jammed channel's inflated variance
pushes its own Mahalanobis distance up, so this doubles as a cheap HDOP-like
indicator). A debounced **boolean** `healthy` flag is still tracked in parallel
(same two-threshold hysteresis as before) purely for dashboard text and arming the
dead-reckoning fallback.

### 2. Confidence-weighted blending (every tick, no threshold to cross)

```
var_gps_eff  = diag(P_gps)  / health_gps          # unhealthy filter's
var_rssi_eff = diag(P_rssi) / health_rssi         # effective covariance inflates

info_gps, info_rssi = 1/var_gps_eff, 1/var_rssi_eff
w_gps   = info_gps / (info_gps + info_rssi)        # per-state inverse-covariance weight
w_rssi  = 1 − w_gps

if health_rssi > RSSI_MIN_HEALTH_FOR_FLOOR:
    w_rssi = max(w_rssi, RSSI_MIN_WEIGHT_FLOOR)    # RSSI always keeps a voice
    w_gps  = 1 − w_rssi

w_gps ← EMA-smoothed against last tick's weight (WEIGHT_SMOOTHING_ALPHA)  # anti-oscillation
x_fused   = w_gps·x_gps + w_rssi·x_rssi
var_fused = 1 / (info_gps + info_rssi)
```

When both filters are perfectly healthy this collapses exactly to plain
inverse-covariance fusion. As one filter degrades, its effective covariance balloons
and it loses influence **continuously** — no threshold decides "how degraded is too
degraded". A **minimum weight floor** for RSSI (`RSSI_MIN_WEIGHT_FLOOR = 0.20`, active once
RSSI's own health exceeds `RSSI_MIN_HEALTH_FOR_FLOOR = 0.35`)
guarantees it always has *some* say whenever it's reasonably healthy, since RSSI is
largely immune to GPS's own attack surface (spoofing/jamming) and can help catch a
slow, subtle GPS spoof before GPS's own gate ever trips.

### 3. Mode label (cosmetic, hysteresis-debounced — never gates the math above)

The blend weights above are *always* continuous. A human-readable label is derived
from them purely for logs/dashboards, with its own vote-counter hysteresis
(`MODE_HYSTERESIS_STEPS = 5`) so it doesn't flicker tick-to-tick:

| Label | Condition |
|---|---|
| 🟦 `GPS-dominant` | `w_gps ≥ MODE_DOMINANT_THRESH` (0.70) |
| 🟧 `RSSI-dominant` | `w_rssi ≥ MODE_DOMINANT_THRESH` (0.70) |
| 🟩 `Balanced Blend` | neither dominant |
| 🟥 `DR` | **both** health scores `< DR_HEALTH_THRESH` (0.22) |

### 4. Dead-reckoning fallback

A shadow state `x_dr` is **continuously propagated with the shared IMU** every tick
(never corrected by any aiding measurement) and is re-synced to the fused solution
whenever the Supervisor is *not* dead-reckoning — so if both filters' health scores
collapse simultaneously, the fallback is already current and takes over with no
discontinuity. Confidence decays linearly the longer the vehicle stays blind,
saturating at 0 after `MAX_DR_TIME_S = 6.0` s.

### 5. Outputs (every IMU tick)

`x_fused` (15-state) + `P_fused`, the debounced `mode` label, a scalar `confidence`
∈ [0,1] (mean of both health scores), the per-tick `w_gps` / `w_rssi` blend weights,
and the raw `health_gps_score` / `health_rssi_score` — all logged so the weight
trajectories themselves can be plotted and audited over a run.

All of the above is configurable from one place, [`config.SupervisorConfig`](config.py)
— rejection/NIS/quality weightings, EMA smoothing rates, the RSSI floor, mode
thresholds, and DR arming — with no changes needed to the fusion logic itself.

---


## Project Structure

```
uav_fault_tolerant_localization/
├── main_supervisor.py         # Main entry point — runs both filters + Supervisor,
│                               #   single-run dashboard, and Monte Carlo studies
├── config.py                  # Every tunable constant: physics, sensor noise,
│                               #   attack parameters, Supervisor thresholds
│
├── filters/
│   ├── ekf_rssi.py             # Algorithm 1: IMU + multi-anchor RSSI EKF
│   ├── ekf_gps.py              # Algorithm 2: IMU + GPS + mag + baro EKF
│   └── supervisor.py           # Supervisor: health monitors, mode logic,
│                               #   blending, dead-reckoning fallback
│
├── models/
│   ├── drone_dynamics.py       # Rotation kinematics, continuous process model,
│                               #   numeric Jacobian, ground-truth trajectories
│   └── sensors.py              # IMU / magnetometer / barometer / RSSI / GPS
│                               #   sensor simulation, incl. attack injection
│
├── utils/
│   ├── monte_carlo.py          # One generic Monte Carlo loop, reused by all
│                               #   three run_monte_carlo() wrappers
│   ├── metrics.py              # Per-run metric formulas (RMSE, consistency,
│                               #   rejection rate, time-to-detect, etc.)
│   └── visualization.py        # Trajectory animation, diagnostic panels,
│                               #   dual-scenario Supervisor dashboard
│
├── logs/                       # Saved Monte Carlo results (CSV/JSON/etc.)
├── results/                    # Saved figures, tables, and plots for the paper/report
└── README.md
```

**Design principle:** every module has exactly one responsibility, and filters call
into shared modules rather than duplicating code — e.g. both `ekf_rssi.py` and
`ekf_gps.py` call `models/drone_dynamics.py` for physics, `models/sensors.py` for
measurement simulation, and `utils/monte_carlo.py` + `utils/metrics.py` for their
Monte Carlo studies.

---

## Getting Started

```bash
pip install numpy matplotlib

# Run everything: both filters + Supervisor, three dashboard windows, diagnostics, Monte Carlo
python main_supervisor.py

# Or run a single filter standalone (single-run animation + diagnostics + its own MC)
python filters/ekf_rssi.py
python filters/ekf_gps.py
```

### Independent per-algorithm attack timing (customization)

The RSSI attack (spoof + NLOS anchors) and the GPS attack (position spoof +
velocity jam) no longer have to start at the same time. Each has its own
config default (`config.RSSI_ATTACK_START_FRACTION`,
`config.GPS_ATTACK_START_FRACTION`, `0.625` / `0.75` of `--duration` by
default) and can be overridden per-run without touching `config.py`. Note this
legacy single-attack timing only applies with `--attack-profile legacy`; the
default `staged` profile uses its own fixed schedule (see
[Main dashboard windows](#main-dashboard-windows)):

```bash
# RSSI attack at t=10s, GPS attack at t=25s, in the SAME 40s run
python main_supervisor.py --rssi-attack-time 10 --gps-attack-time 25

# Same idea expressed as fractions of --duration instead of absolute seconds
python main_supervisor.py --duration 60 --rssi-attack-fraction 0.2 --gps-attack-fraction 0.7

# The standalone single-filter scripts take the equivalent single --attack-time flag
python filters/ekf_rssi.py --duration 40 --attack-time 10
python filters/ekf_gps.py  --duration 40 --attack-time 25

# Headless / no plots (still prints results + runs Monte Carlo)
python main_supervisor.py --no-windows
```

Programmatically (e.g. from a notebook or a sweep script):

```python
from filters import supervisor
from models import drone_dynamics as dyn

sup = supervisor.run_supervised_simulation(
    dyn.true_state_helix, duration=40.0,
    rssi_attack_start_t=10.0,   # RSSI spoof/NLOS kicks in at t=10s
    gps_attack_start_t=25.0,    # GPS spoof/jam kicks in later, at t=25s
)
```

### Main dashboard windows

`main_supervisor.py` now opens **three separate windows**, one per trajectory
scenario:

1. **Linear-Climb** — straight-line climb, mild dynamics.
2. **Helical-Climb** — sustained coordinated turn while climbing.
3. **Figure-8 Aggressive (3D)** — a lemniscate ("figure-8") waypoint pass with
   tighter turns, higher angular rate, and a simultaneous climb, so the drone
   gains altitude *while* threading the figure-8 — the most demanding of the
   three for the coupled attitude/acceleration dynamics.

Each window shows four synchronized, animated **3D** panels — (1) the real
ground-truth trajectory, (2) the IMU+RSSI filter's own prediction, (3) the
IMU+GPS filter's own prediction, and (4) the Supervisor's fused prediction —
with a live, left-aligned status bar underneath reporting **real disruption
numbers** for each filter every tick, not just a healthy/degraded label:
anchors/channels rejected *this update* vs. total checked, the *cumulative*
rejected count since t=0, and each filter's continuous health score (%) —
plus the active fusion mode.

By default `main_supervisor.py` drives a **staged attack profile**
(`config.DEFAULT_ATTACK_PROFILE = "staged"`): three short, escalating attack
windows per run — RSSI-only degradation, then GPS-only degradation, then both
degraded together (to exercise the DR fallback) — instead of one attack that
starts halfway and never ends. Pass `--attack-profile legacy` to fall back to
the single continuous per-sensor attack described under
[Algorithm 1](#algorithm-1--imu--multi-anchor-rssi-ekf) /
[Algorithm 2](#algorithm-2--imu--gps--magnetometer--barometer-ekf) instead.
The per-algorithm diagnostic charts shown after each window closes mark every
attack window's start/end with dotted vertical lines, color-coded by which
sensor(s) were degraded: 🟢 green = RSSI-only, 🟠 orange = GPS-only, 🟣 purple
= both. Windows open **auto-maximized** to fill the screen (tested
at 1920×1080) with a dark, high-contrast color theme (cyan = truth, amber =
IMU+RSSI, violet = IMU+GPS, green = Supervisor fused) — no manual resizing
needed. After each window closes, the familiar "code 1"-style 3×3
error/bias/Mahalanobis diagnostic panels are shown for all three scenarios,
followed by the printed mode-switch/health summary and the Monte Carlo
results.

---

## 📊 Results

All figures, animations, and CSVs below are **auto-generated and reproducible** by
running:

```bash
python Simulation_tests.py
```

which drives the current adaptive Supervisor through a **4 trajectories × 6 attack
scenarios** test matrix (Monte Carlo per cell), saves everything under
[`results/`](results/), and prints the full table to the console. Nothing shown
below is hand-edited — regenerate it any time after touching `config.py` or the
fusion logic to get fresh numbers.

### Scenario matrix — mean RMSE / consistency under the toughest attack

The table below is the **Combined Attack** row (simultaneous GPS spoof+jam *and*
RSSI spoof+NLOS) for each of the four test trajectories — the most stressing single
scenario in the matrix. `%GPS` / `%RSSI` / `%Blend` / `%DR` are the fraction of
run-time the *debounced mode label* spent in each regime (the underlying blend
weights are continuous throughout — see [the Supervisor section](#the-supervisor--context-aware-adaptive-confidence-weighted-fusion)).

| Trajectory | RMSE before (m) | RMSE after (m) | 3σ Consistency | Time-to-detect (s) | %GPS-dom. | %RSSI-dom. | %Blend | %DR | Mean confidence (after) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Straight-Line Climb | 0.40 | 1.18 | 99.1% | 2.92 | 63.3% | 0.0% | 36.7% | 0.0% | 0.50 |
| Helical Turn | 0.38 | 1.09 | 99.8% | 2.47 | 60.6% | 0.0% | 39.4% | 0.0% | 0.49 |
| Hover-Loiter | 0.35 | 1.06 | 99.5% | 2.56 | 60.9% | 0.0% | 39.2% | 0.0% | 0.49 |
| Figure-8 Aggressive | 0.37 | 2.11 | 95.2% | 3.16 | 64.3% | 0.0% | 35.7% | 0.0% | 0.48 |

*(Full 24-row matrix — all 6 scenarios × 4 trajectories, plus GPS/RSSI rejection
rates and mode-switch counts — is saved in
[`results/scenario_matrix_results.csv`](results/scenario_matrix_results.csv).)*

**Reading this table:** even under a simultaneous GPS+RSSI attack, the Supervisor
keeps 3σ covariance consistency near/above 99% on three of the four trajectories
(it dips to ~94% on the aggressive Figure-8, the hardest-to-track path), detects
the attack and starts re-weighting away from the compromised sensor within
2.5–3.2 seconds, and spends the rest of the attack window in a genuine
**Balanced Blend** (~36–39%) rather than snapping to one filter — exactly the
"meaningful RSSI influence even with GPS present" behaviour the adaptive fusion
was designed for.

### Ablation study — does each design choice actually help?

(Helical Turn trajectory, Combined Attack scenario, saved in
[`results/ablation_studies.csv`](results/ablation_studies.csv) /
[`results/ablation_summary.png`](results/ablation_summary.png))

| Ablation | Variant | RMSE after (m) | 3σ Consistency |
|---|---|:---:|:---:|
| **Filter comparison** | Supervisor (fused) | **1.09** | 99.8% |
| | GPS-only | 0.40 | 100.0% |
| | RSSI-only | 1.16 | 98.1% |
| **Gating** | With gating | **1.09** | 99.8% |
| | Without gating | 15.67 | 72.7% |
| **Blending** | With blending (INV_COV) | 1.09 | 99.8% |
| | Without blending (GPS-primary) | 0.94 | 99.7% |

![Ablation summary](results/ablation_summary.png)

Two takeaways jump out: **Mahalanobis gating is doing almost all of the heavy
lifting against outliers** — turn it off and RMSE blows up ~14× — and under this
particular Combined Attack instance plain GPS-primary blending edges out the
adaptive INV_COV blend on raw RMSE, while INV_COV still tracks it closely on 3σ
consistency; the real payoff of the adaptive blend is the graceful, continuous
re-weighting shown in the mode-timeline figures below rather than a guaranteed
RMSE win on every single run (N_MC_RUNS is intentionally kept small here for fast
iteration — see [`Simulation_tests.py`](Simulation_tests.py) to raise it).

### Per-trajectory diagnostic figures (Combined Attack)

Each trajectory below has the same 4-panel diagnostic set, generated in one run of
`Simulation_tests.py`. Click a trajectory to expand its figures.

<details open>
<summary><b>Helical Turn</b> (flagship example — all four panels)</summary>

| Mode timeline | Position error ±3σ |
|---|---|
| ![Helical mode timeline](results/Helical_Turn_mode_timeline.png) | ![Helical pos error 3sigma](results/Helical_Turn_pos_error_3sigma.png) |

| Mahalanobis distance vs. gate | Rejection rate over time |
|---|---|
| ![Helical mahalanobis](results/Helical_Turn_mahalanobis.png) | ![Helical rejection rate](results/Helical_Turn_rejection_rate.png) |

**All 6 attack scenarios, mode timeline comparison** (same trajectory, one row per
scenario — Nominal, GPS Spoofing, GPS Jamming, RSSI Degradation, Combined Attack,
Sudden Total Outage):

![Helical all-scenario mode timeline](results/Helical_Turn_all_scenarios_mode_timeline.png)

</details>

<details>
<summary><b>Straight-Line Climb</b></summary>

| Mode timeline | Position error ±3σ | Mahalanobis vs. gate | Rejection rate |
|---|---|---|---|
| ![](results/Straight-Line_Climb_mode_timeline.png) | ![](results/Straight-Line_Climb_pos_error_3sigma.png) | ![](results/Straight-Line_Climb_mahalanobis.png) | ![](results/Straight-Line_Climb_rejection_rate.png) |

</details>

<details>
<summary><b>Hover-Loiter</b></summary>

| Mode timeline | Position error ±3σ | Mahalanobis vs. gate | Rejection rate |
|---|---|---|---|
| ![](results/Hover-Loiter_mode_timeline.png) | ![](results/Hover-Loiter_pos_error_3sigma.png) | ![](results/Hover-Loiter_mahalanobis.png) | ![](results/Hover-Loiter_rejection_rate.png) |

</details>

<details>
<summary><b>Figure-8 Aggressive</b></summary>

| Mode timeline | Position error ±3σ | Mahalanobis vs. gate | Rejection rate |
|---|---|---|---|
| ![](results/Figure-8_Aggressive_mode_timeline.png) | ![](results/Figure-8_Aggressive_pos_error_3sigma.png) | ![](results/Figure-8_Aggressive_mahalanobis.png) | ![](results/Figure-8_Aggressive_rejection_rate.png) |

</details>

---

## 🎬 Simulation Animations

Live "True vs. Fused" trajectory animations, rendered in full **3D (North-East-Up)**
with a gentle camera pan so climbs and altitude changes are actually visible (not
just the flat North-East plane), under the **Combined Attack** scenario — one per
trajectory type. The fused (green, dashed) track should stay glued to the true
(cyan, solid) track even as the attack kicks in halfway through the run.

<table>
<tr>
<td align="center"><b>Straight-Line Climb</b><br><img src="results/Straight-Line_Climb_dual_trajectory_animation.gif" width="360"></td>
<td align="center"><b>Helical Turn</b><br><img src="results/Helical_Turn_dual_trajectory_animation.gif" width="360"></td>
</tr>
<tr>
<td align="center"><b>Hover-Loiter</b><br><img src="results/Hover-Loiter_dual_trajectory_animation.gif" width="360"></td>
<td align="center"><b>Figure-8 Aggressive (3D climb + pass)</b><br><img src="results/Figure-8_Aggressive_dual_trajectory_animation.gif" width="360"></td>
</tr>
</table>

**Static snapshots** (final frame of each 3D animation above) — useful for a quick
look without loading the full GIF:

<table>
<tr>
<td align="center"><b>Straight-Line Climb</b><br><img src="results/Straight-Line_Climb_dual_trajectory_3d_snapshot.png" width="360"></td>
<td align="center"><b>Helical Turn</b><br><img src="results/Helical_Turn_dual_trajectory_3d_snapshot.png" width="360"></td>
</tr>
<tr>
<td align="center"><b>Hover-Loiter</b><br><img src="results/Hover-Loiter_dual_trajectory_3d_snapshot.png" width="360"></td>
<td align="center"><b>Figure-8 Aggressive (3D climb + pass)</b><br><img src="results/Figure-8_Aggressive_dual_trajectory_3d_snapshot.png" width="360"></td>
</tr>
</table>

For the full **4-panel real-time dashboard** (Real / IMU+RSSI / IMU+GPS /
Supervisor-fused, all in 3D, with a live health/mode status bar, auto-maximized
to your screen), run:

```bash
python main_supervisor.py
```

This now launches **three** dashboard windows — Linear-Climb, Helical-Climb, and
the aggressive 3D Figure-8 climb-and-pass — see
[Main dashboard windows](#getting-started) above, instead of the static GIFs above.

---

## Citation / Acknowledgements

If you use or build on this project (framework, Supervisor design, or the
generated results) in coursework, a thesis, or a paper, a citation or a link back
to this repository is appreciated. Contributions, issues, and pull requests
(new attack models, new trajectories, alternative fusion strategies) are welcome.

**Built for:** resilient multi-sensor UAV navigation research — spoofing/jamming
resistant localization, fault detection & isolation (FDI), and adaptive sensor
fusion under adversarial conditions.
