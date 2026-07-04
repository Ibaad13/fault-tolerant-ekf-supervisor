# UAV Fault-Tolerant Localization — Dual-EKF Sensor Fusion with a Fault-Isolating Supervisor

A 15-state tightly-coupled Extended Kalman Filter (EKF) framework for 3D quadcopter
localization under **contested RF / GPS environments** (spoofing, jamming, NLOS
multipath). Two independent EKFs — one fusing IMU + multi-anchor RSSI ranging, the
other fusing IMU + GPS + magnetometer + barometer — run in parallel under a
**Supervisor** layer that monitors each filter's innovation statistics, isolates
faulty sensors via Mahalanobis gating, and hands off between filters (or falls back
to dead-reckoning) with hysteresis to prevent chattering.

This repository accompanies a study on resilient multi-sensor navigation for UAVs
operating in adversarial or degraded-sensor conditions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Shared 15-State Model](#shared-15-state-model)
3. [Algorithm 1 — IMU + Multi-Anchor RSSI EKF](#algorithm-1--imu--multi-anchor-rssi-ekf)
4. [Algorithm 2 — IMU + GPS + Magnetometer + Barometer EKF](#algorithm-2--imu--gps--magnetometer--barometer-ekf)
5. [The Supervisor — Fault Isolation & Fusion Logic](#the-supervisor--fault-isolation--fusion-logic)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Results](#results)
9. [Simulation Video](#simulation-video)
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

## The Supervisor — Fault Isolation & Fusion Logic

**File:** [`filters/supervisor.py`](filters/supervisor.py)

The Supervisor owns one instance of each EKF, steps both with the **same IMU
sample** every tick (so their internal state trajectories only diverge due to their
different aiding sensors), and fuses their outputs into one navigation solution.

### 1. Per-filter health monitor

Rather than reacting to a single noisy gate decision (one spoofed anchor can coexist
with four clean ones), each filter's *aggregate* gate outcomes are tracked in a
rolling window (`WINDOW_LEN = 20` recent scalar updates):

```
rejection_rate = 1 − mean(accepted flags in window)
```

**Two-threshold hysteresis** avoids chattering right at the boundary:

- Healthy → Degraded when `rejection_rate > 0.35`
- Degraded → Healthy when `rejection_rate < 0.15`

A candidate state must also win **5 consecutive ticks** (`HYSTERESIS_STEPS`) before
the health flag actually flips — a single-sample flicker back across the threshold
does not cause a flip.

### 2. Mode decision table

| GPS healthy | RSSI healthy | Mode | Behaviour |
|:---:|:---:|---|---|
| ✓ | ✓ | `BLEND` | Inverse-variance fusion of both filters |
| ✗ | ✓ | `RSSI` | Use the RSSI filter's state directly |
| ✓ | ✗ | `GPS` | Use the GPS filter's state directly |
| ✗ | ✗ | `DEAD_RECKONING` | Propagate a frozen shadow state with IMU only |

### 3. Blending (`BLEND` mode)

Per-state **inverse-variance (information) weighting**, using `diag(P)` from each
filter as a cheap proxy for the full covariance (valid since `P` stays close to
diagonal after purely sequential scalar updates):

```
info_a = 1 / var_a,     info_b = 1 / var_b
x_fused   = (info_a·x_a + info_b·x_b) / (info_a + info_b)
var_fused = 1 / (info_a + info_b)
```

### 4. Dead-reckoning fallback

A shadow state `x_dr` is **continuously propagated with the shared IMU** every tick
(never corrected by any aiding measurement) and is re-synced to the fused solution
whenever the Supervisor is *not* in `DEAD_RECKONING` mode — so if both filters go
degraded simultaneously, the fallback state is already current and takes over with
no discontinuity. Confidence decays linearly the longer the vehicle stays blind,
saturating at 0 after `MAX_DR_TIME_S = 5.0` s.

### 5. Outputs (every IMU tick)

`x_fused` (15-state), `mode` (`GPS`/`RSSI`/`BLEND`/`DEAD_RECKONING`), a scalar
`confidence` ∈ [0,1], and `active_filter` (which underlying EKF(s) are driving the
fused output).

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

# Run everything: both filters + Supervisor, two dashboard windows, diagnostics, Monte Carlo
python main_supervisor.py

# Or run a single filter standalone (single-run animation + diagnostics + its own MC)
python filters/ekf_rssi.py
python filters/ekf_gps.py
```

### Independent per-algorithm attack timing (customization)

The RSSI attack (spoof + NLOS anchors) and the GPS attack (position spoof +
velocity jam) no longer have to start at the same time. Each has its own
config default (`config.RSSI_ATTACK_START_FRACTION`,
`config.GPS_ATTACK_START_FRACTION`, both 0.5 = halfway through the run by
default) and can be overridden per-run without touching `config.py`:

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

`main_supervisor.py` now opens **two separate windows**, one per trajectory
scenario (Linear-Climb, Helical-Climb). Each window shows four synchronized,
animated 3D panels — (1) the real ground-truth trajectory, (2) the IMU+RSSI
filter's own prediction, (3) the IMU+GPS filter's own prediction, and (4) the
Supervisor's fused prediction — with a live status bar underneath reporting,
in real time, whether IMU+RSSI and/or IMU+GPS are currently
healthy/spoofed/jammed and which fusion mode the Supervisor is in. After each
window closes, the familiar "code 1"-style 3×3 error/bias/Mahalanobis
diagnostic panels are shown for both algorithms, followed by the printed
mode-switch/health summary and the Monte Carlo results.

---

## Results

*(Space reserved for exported figures, tables, and Monte Carlo summaries — see the
[`results/`](results/) folder.)*

<!--
Example layout once populated:

### Single-Run Trajectory Tracking

![Linear climb trajectory](results/linear_trajectory.png)
![Helical climb trajectory](results/helix_trajectory.png)

### Error & Diagnostic Panels

![RSSI diagnostics](results/rssi_diagnostics.png)
![GPS diagnostics](results/gps_diagnostics.png)

### Monte Carlo Summary (30 runs)

| Metric | IMU+RSSI | IMU+GPS | Supervisor |
|---|---|---|---|
| RMSE after attack (m) | | | |
| Rejection rate (attacked) | | | |
| False rejection (clean) | | | |
| Time-to-detect (s) | | | |
| 3σ consistency (%) | | | |

-->

---

## Simulation Video

*(Space reserved for an embedded/linked demo video of the dual-trajectory Supervisor
dashboard in action.)*

<!--
Example once available:

[![Watch the demo](results/video_thumbnail.png)](https://your-video-link-here)

or, if hosted directly in the repo (e.g. via Git LFS):

https://github.com/<user>/<repo>/assets/<video-path>.mp4
-->

---

## Citation / Acknowledgements

*(To be added.)*
