# dashiCFD

CFD + DASHI experiments for 2D vorticity rollouts, ternary structural codecs, and SUSY gauge scans. Assets in the repo are mostly matplotlib outputs; simulations run in pure NumPy with optional Vulkan/vkFFT acceleration.

## Docs
- `docs/overview.md` — quickstart, runner map, and backend notes.
- `docs/signed_filament_annihilation.md` — operator semantics for signed filaments.

## How to Run
- Use Python 3.10+ with `numpy` and `matplotlib`; set `MPLBACKEND=Agg` for headless environments.
- Typical commands:
  - `MPLBACKEND=Agg python dashi_cfd_operator_v3.py`
  - `MPLBACKEND=Agg python dashi_cfd_operator_v4.py`
  - `MPLBACKEND=Agg python dashi_les_vorticity_codec_v2.py`
  - `MPLBACKEND=Agg python vortex_tester_mdl.py`
  - `MPLBACKEND=Agg CORE_BACKEND=cpu python CORE_cfd_operator.py`
  - `MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs`
  - `MPLBACKEND=Agg python perf_kernel.py --z0-npz outputs/kernel_N128_z0.npz --A-npz outputs/kernel_N128_A.npz --steps 20000 --decode-every 200`
  - `MPLBACKEND=Agg python run_les_gpu.py --N 512 --steps 20000 --stats-every 200 --progress-every 2000`

Sprint 51 signed ternary flip audit:

```bash
python3 scripts/ns_signed_ternary_flip_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint51_signed_ternary_flip_gpu_audit
```

Observed six-run N32/N64 result: `NO2CYCLE_FAILS`. Raw cross-shell
minus-to-plus is `93419828142802.9`, plus-to-minus counter-flow is
`84731761817324.95`, and the signed imbalance fraction is
`0.048767829281919015`; the current failing diagnostic is the v1 no-2-cycle
proxy.

Sprint 52 material source / no-2-cycle amplitude audit:

```bash
python3 scripts/ns_sprint52_material_no2cycle_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint52_material_no2cycle_gpu_audit
```

Observed result: `MATERIAL_SOURCE_GATE_CLOSED_NO2CYCLE_AMPLITUDE_BLOCKED`.
Material true-new positive source is absent; the no-2-cycle amplitude proxy
remains the active blocker.

Sprint 53 no-2-cycle physical amplitude audit:

```bash
python3 scripts/ns_sprint53_no2cycle_physical_amplitude_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint53_no2cycle_physical_gpu_audit
```

Observed result: `MATERIAL_SOURCE_GATE_CLOSED_PHYSICAL_NO2CYCLE_AMPLITUDE_BLOCKED`.
Material true-new positive source remains absent, but the material
net-residue physical-amplitude proxy does not clear the sign-cycle gate:
physical-small fraction is `0.3423412506059137` and
`sigma_physical_cycle_fit = -1.1215088689186317`.

Sprint 54 no-2-cycle resolution/cadence audit:

```bash
python3 scripts/ns_sprint54_no2cycle_resolution_cadence_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint54_no2cycle_resolution_cadence_gpu_audit
```

Observed result: `NO2CYCLE_PROXY_OVERCONSERVATIVE_STRETCH_SMALL`.
The Sprint 53 material-mass proxy remains bad, but shell/time direct
`omega dot S omega` evidence reports `small_fraction_by_stretch =
0.9751575375666505`. Cadence remains `single_cadence_unresolved`; packet-local
stretch attribution still requires packet support masks.

Sprint 55 Lagrangian accumulated stretch-action audit:

```bash
python3 scripts/ns_sprint55_lagrangian_stretch_action_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint55_lagrangian_stretch_action_gpu_audit
```

Observed result: `LAGRANGIAN_STRETCH_ACTION_SMALL_DIAGNOSTIC`.
According to Sprint 55, the Sprint 54 stretch diagnostic should be read as
Lagrangian accumulated stretch-action evidence, not as color strings or
packet-color counts. The six-run N32/N64 batch reports
`action_small_fraction = 0.9985242030696576`, `dangerous_lineage_count = 5`,
and `sigma_action_fit = -0.5102412568825301`. It remains diagnostic only:
weighted action summability does not close, cadence and shell-boundary
sensitivity are unresolved, packet-local support masks are unavailable, and all
governance/promotion flags remain false.

Sprint 56 packet-local accumulated stretch-action audit:

```bash
python3 scripts/ns_sprint56_packet_local_stretch_action_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint56_packet_local_stretch_action_gpu_audit
```

Observed result: `PACKET_LOCAL_ACTION_SUMMABILITY_BLOCKED`.
Sprint 56 reconstructs packet-local support masks from Sprint 49 `K_cell`
packet IDs and computes accumulated positive stretch action plus
direction-change separation on those masks. The six-run N32/N64 batch reports
`packet_local_available_fraction = 1.0`, `action_small_fraction =
0.8108028335301063`, `dangerous_lineage_count = 641`, and
`sigma_packet_local_action_fit = -0.4822543927548197`. The packet-local audit
therefore blocks the current accumulated-action route under current
cadence/resolution; all promotion flags remain false.

Sprint 57 vessel/action reconciliation audit:

```bash
python3 scripts/ns_sprint57_vessel_action_reconciliation_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint57_vessel_action_reconciliation_gpu_audit
```

Observed result: `PACKET_ACTION_UNDERCOUNTS_COVERED_STRETCH`.
The six-run N32/N64 batch reports `epsilon_raw_positive_vs_covered =
-0.8161321565334568`, `epsilon_raw_positive_vs_global =
-0.9608719590659198`, and `epsilon_normalized_positive_vs_global =
113.58553013012235`. The current packet-local obstruction is therefore a
normalized packet-action versus raw vessel-action mismatch, not simple
Euclidean packet double-counting; all promotion flags remain false.

Sprint 58 normalized packet-action inflation audit:

```bash
python3 scripts/ns_sprint58_normalized_action_inflation_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint58_normalized_action_inflation_gpu_audit
```

Observed result: `NORMALIZED_ACTION_NONADDITIVE_RATIO_INFLATION`.
The six-run N32/N64 batch reports
`sum_ratios_over_ratio_of_sums_covered = 4904.346096600663`,
`sum_ratios_over_ratio_of_sums_global = 11471.817018880183`, and
`low_enstrophy_denominator_fraction = 0.012394729693018202`. The Sprint 56
packet-normalized `A+` ledger is therefore not vessel-additive; all promotion
flags remain false.

Sprint 59 raw additive packet-stretch audit:

```bash
python3 scripts/ns_sprint59_raw_packet_stretch_action_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint59_raw_packet_stretch_action_gpu_audit
```

This producer computes the vessel-additive packet quantity
`sum_P max(omega dot S omega, 0) dx dt` by voxelwise positive parts, plus
negative/net/raw weighted shell summaries. It is the replacement measurement
for packet-normalized ratio summability; all promotion flags remain false.

Observed six-run N32/N64 result:
`route_decision = RAW_ACTION_SUMMABILITY_BLOCKED`,
`sigma_raw_action_fit = -0.023179743665209647`, and
`A_raw_positive_total = 6825780.534479305`.  This is a near-flat raw-action
spectrum under the current Euclidean `K_cell` packet shelling.  The normalized
packet-ratio obstruction from Sprint 56/58 is no longer the governing object;
the active NS blocker is whether raw positive vortex-stretch action becomes
summable after using an action-preserving cascade geometry.

Current NS planning surface after Sprint 59:

```text
correct additive source:     A_raw_plus = integral max(omega dot S omega, 0) dx dt
current shelling:            Euclidean K_cell packet geometry
current fit:                 sigma_raw_action_fit ~= -0.023
closure threshold:           sigma_raw_action_fit > 0.5
promotion status:            false
next decisive audit:         BT/smoothed shell reassignment with action conservation
```

The next audit must not merely relabel packets.  A corrected shell assignment
is admissible only if it preserves the physical raw-stretch action while
changing the shell distribution:

```text
sum_K A_raw_plus_corrected(K) ~= sum_K A_raw_plus_euclidean(K)
```

The conservation gate must be checked per assignment scheme, per run/time, and
globally:

```text
abs(sum_K A_raw_plus_reassigned(run,time,K) - A_raw_plus_reference(run,time))
  / (abs(A_raw_plus_reference(run,time)) + eps)
  <= conservation_tolerance
```

The same consistency check should be reported for `A_raw_negative` and
`A_raw_net` whenever those ledgers are reassigned.  Smoothed shell windows must
form a partition of unity over each source action contribution; BT/ultrametric
packet addresses must report unassigned and overassigned action fractions.  A
pooled improvement in `sigma_raw_action_fit` is not enough if individual
N/seed/run groups fail conservation or remain flat.

Sprint 60 route decisions should separate conservation failures from real
diagnostic progress:

```text
RAW_ACTION_REASSIGNMENT_SOURCE_UNAVAILABLE
RAW_ACTION_REASSIGNMENT_CONSERVATION_FAILED
RAW_ACTION_REASSIGNMENT_NON_PARTITION_WINDOW_FAILED
RAW_ACTION_REASSIGNMENT_FLAT_BLOCKED
RAW_ACTION_REASSIGNMENT_SUMMABILITY_PROMISING_DIAGNOSTIC
```

If a BT/ultrametric or smoothed shell assignment improves `sigma_raw_action_fit`
while preserving total raw action, the flat Sprint 59 result is evidence of
Euclidean shell-accounting mismatch.  If the fit stays flat under
action-preserving reassignment, the ternary source-budget route remains blocked
and the NS lane should pivot toward direction-coherence/no-concentration
geometry rather than more color-string diagnostics.

BT, p-adic, Monster, and green/zero language is currently an organizing
hypothesis for candidate cascade coordinates, not theorem-grade evidence.  It
becomes proof-relevant only after the reassignment supplies a concrete
action-preserving map and a bounded comparison to the physical raw
`omega dot S omega` ledger.

Sprint 60 raw-action shell reassignment audit:

```bash
python3 scripts/ns_sprint60_raw_action_reassignment_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --raw-action-csv \
    outputs/sprint59_raw_packet_stretch_action_gpu_audit/ns_raw_packet_stretch_action.csv \
  --out-dir outputs/sprint60_raw_action_reassignment_gpu_audit
```

Observed result: `RAW_ACTION_REASSIGNMENT_FLAT_BLOCKED`.  The audit
redistributes the same Sprint 59 raw action ledger through three assignment
schemes and requires per-run/time plus global conservation before interpreting
any shell fit.  All schemes conserve raw action, but all remain far below the
`sigma_raw_action_fit > 0.5` gate:

```text
euclidean_K_cell  sigma = -0.023179743665209647
smoothed_shell    sigma = -0.022555170588204942
bt_ultrametric    sigma = -0.03123173737299591
```

The provisional BT/ultrametric reassignment therefore does not rescue the raw
source-budget route on the current Sprint 49/59 N32/N64 evidence.  The next NS
diagnostic should inspect raw-red lineage anatomy and robustness rather than
assuming shell metric mismatch is the dominant cause.

Sprint 61 raw-red direction-coherence anatomy audit:

```bash
python3 scripts/ns_sprint61_raw_red_direction_coherence_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --raw-action-csv \
    outputs/sprint59_raw_packet_stretch_action_gpu_audit/ns_raw_packet_stretch_action.csv \
  --out-dir outputs/sprint61_raw_red_direction_coherence_gpu_audit
```

Observed result: `RAW_RED_DIRECTION_INCOHERENT_CONCENTRATION_BLOCKED`.
The producer selects the high raw-red population by `weighted_A_raw_positive`,
rebuilds packet masks from Sprint 49 truth, and measures vorticity-direction
coherence, direction-gradient, Beltrami-defect, and parent-lineage metadata.
On the six-run N32/N64 batch:

```text
selected_high_raw_red_count = 1471
incoherent_packet_count = 1451
incoherent_packet_fraction = 0.9864038069340585
direction_coherence_mean_selected = 0.5295134456542706
direction_lipschitz_proxy_mean_selected = 13.970541573972843
```

Under the current packet mask / cadence / coherence-threshold proxy, the high
raw-red packets are not CFM-friendly coherent tubes.  This does not prove a
continuum concentration theorem, but it blocks the immediate DASHI/CFM rescue
of the raw source-budget route on current evidence.  The remaining NS checks
are robustness checks: denser cadence, higher resolution, and a more
theorem-grade direction-coherence proxy.

Sprint 62 direction-coherence robustness audit:

```bash
python3 scripts/ns_sprint62_direction_coherence_robustness_audit.py \
  --sprint61-csv \
    outputs/sprint61_raw_red_direction_coherence_gpu_audit/ns_raw_red_direction_coherence.csv \
  --out-dir outputs/sprint62_direction_coherence_robustness_gpu_audit
```

Observed result:
`DIRECTION_COHERENCE_INCOHERENCE_ROBUST_ON_AVAILABLE_DATA`.  The audit
reclassifies the Sprint 61 packet anatomy across top-population fractions
`0.01, 0.05, 0.10, 0.25, 1.0` and coherence thresholds
`0.60, 0.70, 0.80, 0.90`.  Every tested threshold/top-fraction row remains
incoherent on the available selected population.  By run:

```text
sprint49_material_parent_N64_seed0_gpu: incoherent_fraction = 1.0
sprint49_material_parent_N64_seed1_gpu: incoherent_fraction = 1.0
```

Follow-up GPU ladder runs materialized one higher-resolution and one
dense-cadence check:

```text
N128 seed0, save-every=10:
  Sprint 59 route = RAW_ACTION_SUMMABILITY_BLOCKED
  sigma_raw = -0.015597324361506952
  Sprint 61 route = RAW_RED_DIRECTION_INCOHERENT_CONCENTRATION_BLOCKED
  Sprint 62 route = DIRECTION_COHERENCE_INCOHERENCE_ROBUST_ON_AVAILABLE_DATA

N64 seed0, save-every=2:
  Sprint 59 route = RAW_ACTION_SUMMABILITY_BLOCKED
  sigma_raw = -0.03440204529464993
  Sprint 61 route = RAW_RED_DIRECTION_INCOHERENT_CONCENTRATION_BLOCKED
  Sprint 62 route = DIRECTION_COHERENCE_INCOHERENCE_ROBUST_ON_AVAILABLE_DATA
```

The N128 and dense-cadence checks strengthen the current negative diagnostic,
but they still do not prove a continuum concentration theorem. N1024-scale
evidence and a theorem-grade direction-coherence proxy remain open. Sprint 61
now caches frame-wide direction fields/gradients once per saved frame, so the
coherence pass is no longer the dominant runtime; GPU work should prioritize
truth generation, spectral derivatives, and compact packet-bin accumulation.

Sprint 63 cross-shell replenishment contractivity audit:

```bash
python3 scripts/ns_sprint63_cross_shell_replenishment_contractivity_audit.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --raw-action-csv \
    outputs/sprint59_raw_packet_stretch_action_gpu_audit/ns_raw_packet_stretch_action.csv \
  --out-dir outputs/sprint63_cross_shell_replenishment_contractivity_gpu_audit
```

Sprint 63 is the next DASHI-native theorem fork after Sprint 60 shell
reassignment and Sprint 61/62 direction coherence both stayed blocked.  It
joins Sprint 49 material parent edges to Sprint 59 raw-action packet ledgers
and asks whether cross-shell or adjacent-shell replenishment is non-amplifying:
the child packet's raw positive action is compared with the parent packet's
available raw positive action, scaled by the Sprint 49 credited-mass fraction.

The producer reports edge, by-`K`, and by-transition-state surfaces:

```text
parent_action_budget = A_raw_positive(parent) * credited_mass / parent_mass
contractivity_ratio  = A_raw_positive(child) / (parent_action_budget + eps)
```

Rows with `contractivity_ratio <= 1` are contractive under this diagnostic
ledger.  Rows with ratio above one are replenishment-amplifying and block the
candidate theorem unless they can be discharged by a stronger admissibility or
defect argument.  The script preserves multi-parent Sprint 49 edges directly;
it does not use Sprint 61's selected high-red one-row parent map.

Route decisions:

```text
CROSS_SHELL_REPLENISHMENT_SOURCE_UNAVAILABLE
CROSS_SHELL_REPLENISHMENT_NO_EDGES
CROSS_SHELL_REPLENISHMENT_CONTRACTIVE_ON_AVAILABLE_DATA
CROSS_SHELL_REPLENISHMENT_MIXED
CROSS_SHELL_REPLENISHMENT_NONCONTRACTIVE_BLOCKED
```

This is a diagnostic surface for the formal target
`AdjacentCrossShellReplenishmentSummable`: support non-creation plus admissible
parent transfer plus defect/action non-amplification should imply that
cross-shell parent credit cannot sustain unbounded raw-red action.  Sprint 63
does not prove support non-creation, defect monotonicity, stretch absorption,
no finite-time blowup, or any Clay/NS promotion.

Observed six-run N32/N64 result:
`CROSS_SHELL_REPLENISHMENT_MIXED`, but strongly noncontractive under the raw
parent-budget diagnostic:

```text
edge_count = 26150
available_parent_action_edge_count = 25505
contractive_edge_count = 1192
noncontractive_edge_fraction = 0.9532640658694373
contractivity_ratio_total = 2.7665497780287076
weighted_contractivity_ratio_total = 2.9828906939689044
support_created_fraction_proxy_mean = 0.603752604523159
```

The N128 and dense-cadence follow-ups agree:

```text
N128 seed0:
  route = CROSS_SHELL_REPLENISHMENT_MIXED
  noncontractive_edge_fraction = 0.9987681013676589
  contractivity_ratio_total = 4.371227592340793
  weighted_contractivity_ratio_total = 5.806885413286424

N64 seed0, save-every=2:
  route = CROSS_SHELL_REPLENISHMENT_MIXED
  noncontractive_edge_fraction = 0.9901806026277914
  contractivity_ratio_total = 2.6548745195597747
  weighted_contractivity_ratio_total = 2.8200893072403987
```

This blocks the simple raw-action parent-credit contractivity theorem on
current artifacts.  A future formal proof would need a stronger defect norm,
support non-creation hypothesis, or admissibility quotient than the raw
positive-action budget used here.  Without that stronger structure, the current
DASHI NS source-budget route is diagnostically exhausted; the remaining
proof-facing paths are a different CFM/BKM/concentration-compactness bridge or
the independent YM lane.

Post-Sprint 63 NS verdict:

```text
NS source-budget route: DIAGNOSTICALLY_EXHAUSTED_ON_CURRENT_ARTIFACTS
  normalized packet action: non-additive, Jensen-inflated
  raw action shell summability: flat, sigma ~= -0.02 to -0.03
  BT/smoothed reassignment: action-preserving but still flat
  CFM direction-coherence proxy: incoherent on N64, dense N64, N128
  cross-shell parent-budget contractivity: noncontractive, 2.7x-4.4x total ratio
```

The next NS route is therefore a norm switch rather than another
color-string/shell/action-budget diagnostic.  Sprint 64 aligns the DASHI
`Overflow` surface with the CKN/ESS critical concentration picture:

```text
grounded = local scale-critical concentration below diagnostic epsilon
plateau  = near diagnostic epsilon
ascended = above diagnostic epsilon / candidate concentration site
```

Current truth artifacts contain `velocity_snapshots` but no pressure field.
Sprint 64 should therefore emit a velocity-only local `L3` concentration audit
and explicitly route as pressure-reconstruction-missing until the
`|p|^(3/2)` term is available.  This is a CKN-aligned diagnostic surface only;
it is not a CKN epsilon-regularity certificate and carries no Clay/NS
promotion.

Sprint 64 CKN/local critical concentration audit:

```bash
python3 scripts/ns_sprint64_ckn_local_critical_concentration_audit.py \
  --inputs \
    outputs/truth3d/ns3d_N32_seed0_gpu.npz \
    outputs/truth3d/ns3d_N32_seed1_gpu.npz \
    outputs/truth3d/ns3d_N32_seed2_gpu.npz \
    outputs/truth3d/ns3d_N32_seed3_gpu.npz \
    outputs/truth3d/ns3d_N64_seed0_gpu.npz \
    outputs/truth3d/ns3d_N64_seed1_gpu.npz \
  --out-dir outputs/sprint64_ckn_local_critical_concentration_gpu_audit \
  --scales 8 16 \
  --epsilon-critical 0.01 \
  --plateau-fraction 0.5
```

Observed six-run N32/N64 result:
`LOCAL_CRITICAL_CONCENTRATION_PRESSURE_RECONSTRUCTION_MISSING`.
The audit produced 18720 velocity-only parabolic block rows, with
`ascended_fraction = 0.9642628205128205`,
`max_local_critical_quantity = 0.5370477381890987`, and
`max_local_concentration_ratio = 53.704773818909864` under the diagnostic
`epsilon_critical = 0.01`.

The N128 and dense-cadence follow-ups also route as pressure-missing:

```text
N128 seed0:
  row_count = 59904
  ascended_fraction = 0.8740818643162394
  max_local_concentration_ratio = 10.67887709906586

N64 seed0, save-every=2:
  row_count = 35136
  ascended_fraction = 0.926542577413479
  max_local_concentration_ratio = 24.25368457905771
```

These numbers are not a CKN verdict.  They say only that the velocity-side
critical concentration surface is now measurable and currently nontrivial, but
the full CKN quantity is unavailable until pressure is reconstructed or stored.

Sprint 65 pressure reconstruction target:

```text
Delta p = - sum_ij (partial_i u_j) (partial_j u_i)
mean(p) = 0 per periodic frame
```

Sprint 65 should append `pressure_snapshots` to pressure-augmented truth NPZs
and report Poisson residual/gauge diagnostics before rerunning Sprint 64 with
the pressure term present.  This still does not promote CKN regularity; it only
removes the artifact-level `pressure_reconstruction_missing` blocker.

Sprint 65 pressure reconstruction:

```bash
python3 scripts/ns_sprint65_pressure_reconstruction.py \
  --inputs \
    outputs/truth3d/ns3d_N32_seed0_gpu.npz \
    outputs/truth3d/ns3d_N32_seed1_gpu.npz \
    outputs/truth3d/ns3d_N32_seed2_gpu.npz \
    outputs/truth3d/ns3d_N32_seed3_gpu.npz \
    outputs/truth3d/ns3d_N64_seed0_gpu.npz \
    outputs/truth3d/ns3d_N64_seed1_gpu.npz \
  --out-dir outputs/sprint65_pressure_reconstruction_gpu_audit \
  --overwrite
```

Observed six-run N32/N64 result:

```text
max_poisson_relative_residual_rms = 3.5409688067143674e-16
pressure_gauge = zero_mean_per_frame
promotion_status = NO_PROMOTION_SPRINT65_PRESSURE_RECONSTRUCTION_DIAGNOSTIC
```

The pressure-present Sprint 64 rerun advances beyond the missing-pressure
route:

```bash
python3 scripts/ns_sprint64_ckn_local_critical_concentration_audit.py \
  --inputs \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed1_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed2_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed3_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed1_gpu_pressure.npz \
  --out-dir outputs/sprint64_ckn_local_critical_concentration_pressure_gpu_audit \
  --scales 8 16 \
  --epsilon-critical 0.01 \
  --plateau-fraction 0.5
```

Observed pressure-present route:

```text
route_decision = LOCAL_CRITICAL_CONCENTRATION_MIXED
row_count = 18720
ascended_fraction = 0.9890491452991453
max_local_concentration_ratio = 60.83081878566949
```

N128 seed0 and dense N64 seed0 also route as mixed:

```text
N128 seed0:
  ascended_fraction = 0.9127771100427351
  max_local_concentration_ratio = 12.148072897848532

N64 seed0, save-every=2:
  ascended_fraction = 0.9361338797814208
  max_local_concentration_ratio = 27.578561542239555
```

Sprint 65 therefore removes the artifact-level pressure blocker, but it does
not close the CKN route.  The next CKN-facing gate is threshold/proxy
calibration and theorem-grade interpretation of the pressure-inclusive local
concentration surface.

Sprint 66 CKN r-sweep calibration target:

```text
C(r, x0, t0) = r^-2 integral_Q(r,x0,t0) (|u|^3 + |p|^(3/2)) dx dt
```

The Sprint 64/65 fixed-block ascended fractions should not be read as a
near-singularity verdict.  Sprint 66 should sample candidate hot spots and
track the scale-normalized pressure-inclusive quantity over several radii. The
diagnostic question is whether `C(r)` decays, stays flat, or increases as the
audit zooms inward around the hottest packet/field centres.

Sprint 66 CKN r-sweep calibration:

```bash
python3 scripts/ns_sprint66_ckn_r_sweep_calibration.py \
  --inputs \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed1_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed2_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed3_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed1_gpu_pressure.npz \
  --candidate-csv outputs/sprint59_raw_packet_stretch_action_gpu_audit/ns_raw_packet_stretch_action.csv \
  --out-dir outputs/sprint66_ckn_r_sweep_calibration_gpu_audit \
  --r-cells 2 4 8 16 \
  --epsilon-grid 0.01 0.05 0.1 0.5 1.0 \
  --top-hotspots 10
```

Route decisions:

```text
CKN_R_SWEEP_SOURCE_UNAVAILABLE
CKN_R_SWEEP_NO_HOTSPOTS
CKN_R_SWEEP_PRESSURE_RECONSTRUCTION_MISSING
CKN_R_SWEEP_SUBCRITICAL_ON_SAMPLED_HOTSPOTS
CKN_R_SWEEP_DECAYS_UNDER_ZOOM
CKN_R_SWEEP_MIXED
CKN_R_SWEEP_CRITICAL_BLOCKED
```

This remains a DNS/proxy calibration surface.  It does not apply a CKN
epsilon-regularity theorem, prove a suitable weak solution bridge, establish
continuum uniformity, prove no finite-time blowup, or promote Clay/NS.

Observed Sprint 66 results:

```text
six-run N32/N64:
  route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM
  row_count = 1200
  hotspot_count = 60
  ascended_fraction = 0.43666666666666665
  decaying_hotspot_count = 60
  concentrating_hotspot_count = 0
  mean_log_slope_dlogC_dlogr = 0.8891490092755248

N128 seed0:
  route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM
  row_count = 250
  hotspot_count = 10
  ascended_fraction = 0.116
  decaying_hotspot_count = 10
  concentrating_hotspot_count = 0
  mean_log_slope_dlogC_dlogr = 1.2761705831944667

N64 seed0, save-every=2:
  route_decision = CKN_R_SWEEP_DECAYS_UNDER_ZOOM
  row_count = 200
  hotspot_count = 10
  ascended_fraction = 0.125
  decaying_hotspot_count = 10
  concentrating_hotspot_count = 0
  mean_log_slope_dlogC_dlogr = 1.2046696180516625
```

Sprint 67B CKN uniformity audit:

```bash
python3 scripts/ns_sprint67_ckn_uniformity_audit.py \
  --inputs \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed1_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed2_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N32_seed3_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed0_gpu_pressure.npz \
    outputs/sprint65_pressure_reconstruction_gpu_audit/ns3d_N64_seed1_gpu_pressure.npz \
  --sprint64-csv outputs/sprint64_ckn_local_critical_concentration_pressure_gpu_audit/ns_local_critical_concentration.csv \
  --out-dir outputs/sprint67_ckn_uniformity_gpu_audit \
  --r-cells 2 4 8 16 \
  --epsilon-critical 0.01
```

Sprint 67B removes the top-hotspot selection bias by replaying fixed-block
Sprint 64 ascended cylinders, computing pressure-inclusive `C(r)` r-sweeps at
each candidate centre, and clustering adjacent ascended blocks.

Route decisions:

```text
CKN_UNIFORM_DECAY_SUPPORTED
CKN_LOCALIZED_PERSISTENT_PLATEAU
CKN_CONCENTRATION_CANDIDATE_FOUND
CKN_PRESSURE_DOMINATED_ARTIFACT
CKN_INCONCLUSIVE_NEEDS_HIGHER_N
CKN_UNIFORMITY_SOURCE_UNAVAILABLE
```

This remains a DNS/proxy uniformity surface. It does not apply CKN epsilon
regularity, prove suitable weak-solution status, establish continuum
uniformity, prove no finite-time blowup, or promote Clay/NS.

Observed bounded six-run Sprint 67B result:

```text
route_decision = CKN_UNIFORM_DECAY_SUPPORTED
cylinder_count = 1536
cluster_count = 120
decaying_count = 1536
flat_count = 0
concentrating_count = 0
persistent_cluster_count = 0
pressure_fraction_max = 0.13074814940071125
max_C_total_N32 = 0.6157542190448191
max_C_total_N64 = 0.2939492011581624
max_ckn_grows_with_N = false
```

These r-sweeps demote the fixed-block Sprint 64/65 ascended fractions: sampled
hot spots decay under zoom rather than concentrating.  This is favorable
CKN-aligned diagnostic evidence for the available DNS artifacts, not a
theorem-grade CKN certificate.

R/G/B packet-thread volume visualization:

```bash
python3 scripts/ns_rgb_thread_volume_visualizer.py \
  --input outputs/sprint49_material_parent_N64_seed0_gpu \
  --out-dir outputs/rgb_thread_volume_visuals \
  --time latest \
  --alpha 0.30 \
  --max-points 80000 \
  --background transparent
```

The visualizer reconstructs Sprint 49 `K_cell` packet masks, stores empty/nil
cells as `NaN` in the exported volume, and renders plus/red, zero/green, and
minus/blue packet voxels at the requested opacity.  Current Sprint 49 child
state tables contain plus/minus but no zero rows, so green is supported by the
tool but absent in the generated six-run child-state images.

To animate the R/G/B packet volume over all available material-parent times:

```bash
python3 scripts/ns_rgb_thread_volume_visualizer.py \
  --input outputs/sprint49_material_parent_N64_seed0_gpu \
  --out-dir outputs/rgb_thread_volume_visuals \
  --all-times \
  --animation-format both \
  --fps 6 \
  --alpha 0.30 \
  --max-points 80000 \
  --background transparent
```

After Sprint 59, action-derived trits can be visualized with:

```bash
python3 scripts/ns_rgb_thread_volume_visualizer.py \
  --input outputs/sprint49_material_parent_N64_seed0_gpu \
  --out-dir outputs/rgb_thread_volume_visuals \
  --all-times \
  --trit-source raw_action \
  --raw-action-csv outputs/sprint59_raw_packet_stretch_action_gpu_audit/ns_raw_packet_stretch_action.csv \
  --raw-action-threshold 50 \
  --animation-format both \
  --alpha 0.30
```

## Vulkan GPU commands

Compile SPIR-V (preferred path `dashiCORE/spv/comp` -> `dashiCORE/spv`):

```bash
python dashiCORE/scripts/compile_spv.py
```

Kernel-only perf (GPU rollout + Vulkan decode, metrics-only readback):

```bash
python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --decode-every 200 \
  --decode-backend vulkan \
  --observer metrics \
  --backend vulkan \
  --fft-backend vkfft-vulkan \
  --op-backend vulkan \
  --require-gpu \
  --metrics-json outputs/perf_metrics_gpu.json
```

Kernel-only perf (full ω̂ readback for energy/enstrophy):

```bash
python perf_kernel.py \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --decode-every 200 \
  --decode-backend vulkan \
  --observer snapshots \
  --backend vulkan \
  --fft-backend vkfft-vulkan \
  --op-backend vulkan \
  --require-gpu \
  --metrics-json outputs/perf_snapshots_gpu.json
```

Long kernel-only GPU run with visuals:

```bash
MPLBACKEND=Agg python run_v4_snapshots.py \
  --kernel-only \
  --z0-npz outputs/kernel_N128_z0.npz \
  --A-npz outputs/kernel_N128_A.npz \
  --steps 20000 \
  --stride 200 \
  --no-ground-truth \
  --out-dir outputs \
  --prefix kernel_N128 \
  --backend vulkan \
  --op-backend vulkan \
  --decode-backend vulkan \
  --fft-backend vkfft-vulkan \
  --timing \
  --progress-every 200
```

Sprint 49 3D GPU truth + material-parent audit:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/make_truth_3d.py \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --N 64 \
  --steps 120 \
  --save-every 10 \
  --dt 0.001 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N64_seed0_gpu.npz \
  --progress-every 10

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/ns_material_parent_summary.py \
  --truth outputs/truth3d/ns3d_N64_seed0_gpu.npz \
  --out-dir outputs/sprint49_material_parent_N64_seed0_gpu \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --diagnostic-precision float64 \
  --progress-every 4
```

Current Sprint 49 batch summary:

- `outputs/sprint49_material_parent_gpu_batch/sprint49_material_parent_gpu_batch_summary.csv`
- `outputs/sprint49_material_parent_gpu_batch/sprint49_material_parent_gpu_batch_summary.json`

Sprint 50 full ternary cross-shell audit from existing Sprint 49 artifacts:

```bash
python3 scripts/ns_ternary_cross_shell_matrix.py \
  --inputs \
    outputs/sprint49_material_parent_N32_seed0_gpu \
    outputs/sprint49_material_parent_N32_seed1_gpu \
    outputs/sprint49_material_parent_N32_seed2_gpu \
    outputs/sprint49_material_parent_N32_seed3_gpu \
    outputs/sprint49_material_parent_N64_seed0_gpu \
    outputs/sprint49_material_parent_N64_seed1_gpu \
  --out-dir outputs/sprint50_full_ternary_cross_shell_gpu_audit
```

Current Sprint 50 batch routes as `CROSS_PLUS_FROM_MINUS_DOMINATES`; the
producer uses `parent_state -> child_state` and derives source kind from
`parent_relation` plus shell delta, not from Sprint 49 `classification`.

The current material-parent backend is
`gpu_spectral_gradient_cpu_packets`: fp64 Vulkan/vkFFT spectral derivatives are
computed on GPU, while packet matching and bin reduction are still CPU-side.
The next performance step is GPU packet-bin accumulation to avoid full
derivative readback.

Enstrophy graph from the snapshots metrics JSON:

```bash
python scripts/plot_enstrophy.py \
  --input outputs/perf_snapshots_gpu.json \
  --output outputs/enstrophy_kernel_only.png
```

GPU-only LES run (vkFFT + Vulkan, enstrophy CSV + optional PNGs):

```bash
MPLBACKEND=Agg python run_les_gpu.py \
  --N 512 \
  --steps 20000 \
  --dt 0.01 \
  --nu0 1e-4 \
  --Cs 0.17 \
  --stats-every 200 \
  --progress-every 2000 \
  --viz-every 2000 \
  --spectral-truncation exp \
  --trunc-alpha 36 \
  --trunc-power 8 \
  --out-dir outputs \
  --prefix les_gpu
```

Enstrophy plot from the LES CSV:

```bash
python scripts/plot_enstrophy.py \
  --input outputs/les_gpu_enstrophy.csv \
  --output outputs/enstrophy_les_gpu.png \
  --format csv \
  --title "LES GPU enstrophy"
```

NS->EV5 shell-enstrophy adapter (evidence-only projection diagnostics):

```bash
PYTHONPATH=. python scripts/make_truth.py \
  --backend cpu \
  --N 64 \
  --steps 100 \
  --stride 10 \
  --out outputs/truth/ns_ev5

python scripts/ns_ev5_shell_enstrophy.py \
  --truth outputs/truth/ns_ev5_cpu_YYYY-MM-DDTHHMMSS.npz \
  --out-dir outputs/ns_ev5_probe \
  --lane-mode ns-ev5 \
  --burn-in 1 \
  --q-tolerance 0 \
  --cycle-window 8
```

3D periodic incompressible truth artifact for physical bridge falsification:

```bash
python3 scripts/make_truth_3d.py \
  --N 32 \
  --steps 200 \
  --save-every 10 \
  --dt 0.002 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N32_seed0.npz
```

Opt-in SPV/vkFFT lane using vendored `dashiCORE` Vulkan infrastructure:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 -u scripts/make_truth_3d.py \
  --backend gpu \
  --fft-backend vkfft-vulkan \
  --N 32 \
  --steps 200 \
  --save-every 10 \
  --dt 0.002 \
  --nu0 0.001 \
  --seed 0 \
  --out outputs/truth3d/ns3d_N32_seed0_gpu.npz
```

Use the ICD path that exists on the host (`/usr/share/vulkan/icd.d/radeon_icd.json`
on this machine; some older notes use `radeon_icd.x86_64.json`). On the RX
580/gfx803 host, first verify the compatibility environment if Vulkan or ROCm
visibility is uncertain:

```bash
cd /home/c/Documents/code/__OTHER/gfx803_compat_graph/gfx803_flake_v1
nix develop .#base
verify-gfx803-host
```

The 3D truth lane is separate from the existing 2D DASHI/LES pipeline. It
writes `omega_snapshots` with shape `(T,N,N,N,3)`, `velocity_snapshots` with
the same vector shape by default, `steps`, diagnostics, and `meta_json`
declaring `dimension=3`, periodic boundaries, Leray projection, and 2/3
dealiasing. The CPU path is the default; the GPU path is fail-fast and records
the vkFFT backend, Vulkan device info, and 3D SPV shader list. The script fails
closed when saved fields are non-finite,
divergence/curl checks fail, CFL exceeds the configured bound, or fewer than
five vorticity shells are nonzero at or above `K_star`. GPU validation uses
fp32-appropriate divergence/curl tolerance floors and prints the actual
`fft_plan_backend` plus Vulkan device name so CPU/GPU execution is auditable.
Packet lineage labels are intentionally deferred until after the 3D physical
bridge harness runs.

FP64 GPU parity probe for diagnostics:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
python3 scripts/probe_vkfft_fp64.py --N 16
```

On the RX580/RADV host this reports `shaderFloat64=true` and complex128 vkFFT
round-trip error around `1e-15`. The fp64 path is a diagnostic/parity lane, not
the default speed lane; the 3D truth solver still defaults to fp32 GPU kernels.

No truth NPZ is required for CLI validation:

```bash
python scripts/ns_ev5_shell_enstrophy.py \
  --smoke \
  --out-dir outputs/ns_ev5_smoke
```

Residual atom codec probe (phase-aware signed anisotropic atoms; empirical
frame-bound test only):

```bash
python scripts/residual_atom_codec_probe.py \
  --truth outputs/truth_ns_ev5_cpu_2026-06-01T014101.npz \
  --snapshot-index -1 \
  --out outputs/atom_codec_probe_ns_ev5.json \
  --plot outputs/atom_codec_probe_ns_ev5.png \
  --max-atoms 16 \
  --peak-candidates 16 \
  --q 0.05
```

The probe writes selected atom fields, reconstruction metrics, and selected
Gram/frame diagnostics. It replaces random-phase residual synthesis with a
deterministic signed atom search for this experiment, but it is not a Gate3
proof, NS regularity theorem, or production codec replacement. Its selected
lower Gram eigenvalue is only an empirical `A > 0` probe for the
`AtomExtendedCarrierFrameReceipt` frame-bound obligation; it is not a uniform
lower frame-bound proof or continuum norm-comparison theorem.

The adapter writes `manifest.json`, `shell_enstrophy.csv`, `ev5_trace.csv`,
`theta_profile.csv`, and `checks.json`. The default `--lane-mode ns-ev5` uses the corrected
diagnostic dictionary: lane 2 is the bucketed enstrophy-weighted mean shell,
lane 3 is an adjacent cascade-flux ratio diagnostic only, lane 5 is secondary
top-shell occupancy, lane 7 is the dissipative enstrophy tail above `K*(nu)`,
and lane 11 is a `Z/3` phase/coherence proxy outside canonical FRACTRAN rules.
`Q_log` is still emitted as `lane2 + lane7`, but it is retained only as a
falsified scalar diagnostic from earlier checks, not as the live descent
criterion. The live vector EV5 check is lane 7 non-increasing together with
`mean_shell <= K_star + 1`. Lane 2 is a coordinate/boundedness witness rather
than Lyapunov energy, and lane 3 remains an adjacent cascade-flux diagnostic
excluded from Lyapunov/descent logic. Use `--lane-mode legacy-jmax-peak` only
to compare against the former `j_max` / peak-shell lane mapping.

For the corrected lane 7, the adapter reads `nu0`, `nu`, or `viscosity` from
`meta_json`, or accepts `--nu` / `--tail-k-star`. If the normalization guard
fails, `checks.json` fails closed instead of suppressing the falsification.
The theta runtime diagnostic is a finite cutoff/time vector:
`theta(k,t)=|Flux_tail(k,t)|/Diss_tail(k,t)` for fixed cutoffs `k >= K*(nu)`,
`Theta=max_k sup_t theta(k,t)`, and `danger_shell` is the argmax over that
fixed cutoff profile. It assumes no monotonicity in `k`; missing or zero
dissipation fails closed.
Finite-window EV5 cycles are flagged in a single trace, but become falsifying
resonant-cycle evidence only with a forced vs no-forcing comparison manifest.
These outputs are empirical diagnostics only; they do not discharge NS->EV5,
Gate3 norm, or Clay obligations.

Interactive CLI (recommended for day-to-day runs):

```bash
python dashi_cli.py les --interactive
python dashi_cli.py kernel --interactive
python dashi_cli.py plot --interactive
python dashi_cli.py compare --interactive
```

## Runners: which one is canonical?

Use `run_v4_snapshots.py` as the canonical runner for end-to-end DASHI CFD experiments (LES → encode → learn → rollout → decode + plots/metrics). It is the only script that exercises the full v4 pipeline with consistent CLI flags for backends, FFT selection, timing, and optional GPU encode/decode.

Other runners are targeted utilities:
- `run_les_gpu.py` — GPU LES only (no DASHI encode/learn/rollout); use it for LES stability/perf sweeps or to generate standalone LES diagnostics.
- `perf_kernel.py` — kernel-only (A/z) rollout + decode for performance/metrics; use it for backend micro-benchmarking and metrics-only runs.
- `dashi_cfd_operator_v3.py` / `dashi_cfd_operator_v4.py` — pure Python module demos; keep for reference/quick sanity runs but do not treat them as the primary CLI.
- `dashi_cli.py` — convenience wrapper that dispatches to the scripts above; `run_v4_snapshots.py` remains the canonical path under the hood for full runs.

Rule of thumb: if you want a result you’ll carry forward or compare across backends, run `run_v4_snapshots.py`.

## Latest Run Results (last recorded 2026-01-24, headless)
- `dashi_cfd_operator_v3.py` — success; baseline 300 steps in 0.619s (2.06 ms/step). Final relL2 0.473, corr 0.881, ΔE −1.221e-03, ΔZ −1.077e-01.
- `dashi_cfd_operator_v4.py` — success; baseline 300 steps in 0.627s (2.09 ms/step). Final relL2 0.648, corr 0.787, ΔE −2.38e-04, ΔZ −1.64e-02. Now preserves top-128 mid-band phases (indices fixed) and only synthesizes the remaining mid/high energy.
- `dashi_les_vorticity_codec.py` — success; codec stats: compression_ratio 0.714, relL2 0.03997, corr 0.9992, support_cells 4078.
- `dashi_les_vorticity_codec_v2.py` — success; sim 1.114s, codec 0.016s. q sweep: ratios 4.10→6.95, relL2 0.091→0.106, corr 0.996→0.994.
- `vortex_tester_mdl.py` — ran; only FigureCanvasAgg warnings (plots suppressed).
- `naw.py` — failed at `plot_3d_isosurface_voxels`: ValueError broadcasting (10,60,60) vs (11,61,61) after completing 10 g3 slices; also log10 divide-by-zero warnings.
- `naw2.py` — failed: `ModuleNotFoundError: No module named 'skimage'` (needs `scikit-image` for marching_cubes).
- `CORE_cfd_operator.py` — compares legacy numpy gating vs dashiCORE Carrier path. N=64, steps=120: legacy 3.05 ms/step; core (accelerated) 3.35 ms/step; core+fused mask 3.10 ms/step. Larger grids with fused mask: N=128 → legacy 8.21 ms/step vs core 8.18 (speedup 1.004×); N=256 → legacy 32.42 vs core 32.65 ms/step (speedup 0.993×). Enstrophy matches across runs; mask_ops≈0 on accelerated paths.
- `CORE_cfd_operator.py` @ 1024×1024, steps=120 (accelerated, fused mask) — legacy path not rerun; core fused path: 702.2 ms/step, enstrophy 0.00629, mask_mean 0.916. Final vorticity snapshot saved to `outputs/core_1024_final.png`.
- `outputs/v4_t300_compare.png` — side-by-side ω true / decoded+residual / error at t=300 (N=64) generated from `dashi_cfd_operator_v4.py` pipeline.
- `run_v4_snapshots.py` — CLI runner to save triptychs every stride: e.g., `MPLBACKEND=Agg python run_v4_snapshots.py --N 64 --steps 3000 --stride 300 --out-dir outputs --dpi 150 --figsize 14,5 --progress-every 100`. Supports `--pix-width/--pix-height` for exact pixels, `--traj-npz` to reuse a stored trajectory, `--save-traj` to write one, `--no-ground-truth` to skip ω_true/error panels (must pair with `--traj-npz`), `--timing` to print stage timings, `--dtype {auto,float32,float64}` (auto → float64 when N>1024), `--backend {cpu,accelerated,vulkan}` (best-effort; falls back to CPU if unavailable), and `--fft-backend {numpy,vkfft,vkfft-opencl,vkfft-vulkan}`. Vulkan path now tries vkFFT for FFTs (and falls back to NumPy if bindings/ICD missing). To force a specific ICD: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MPLBACKEND=Agg python run_v4_snapshots.py ... --backend vulkan --fft-backend vkfft-vulkan`. Kernel-only start: `--kernel-only --z0-npz path.npz` where the file contains `z` (proxy vector), `mask_low` (bool mask), `anchor_idx` (int indices for preserved mid-band coeffs).
- `run_v4_snapshots.py` decode parity options: `--check-decode-parity` compares CPU vs Vulkan decode at snapshot steps and logs rel-L2/corr (plus low-pass/mask parity). `--parity-only` runs the parity check without plotting or ground truth; it implies `--check-decode-parity` and `--no-ground-truth`. When `--decode-backend vulkan` is requested, the runner now fails fast if GPU decode was not used (no silent CPU fallback).
- Vulkan decode semantics: annihilation now incorporates lifetime when deciding survival (long-lived support gets a slightly relaxed coherence threshold). This is a behavioral choice; rebuild SPIR-V after editing shaders via `python dashiCORE/scripts/compile_spv.py`.
- `run_v4_snapshots.py` also accepts `--les-backend {cpu,gpu}` to generate ground-truth LES on GPU via `VulkanLESBackend` (still reads back each step for CPU-side encoding).
- When using `--les-backend gpu`, you can enable spectral truncation with `--spectral-truncation exp --trunc-alpha 36 --trunc-power 8`.
- `run_v4_snapshots.py` supports `--encode-backend gpu` to run the encode path on GPU; the first step bootstraps `anchor_idx` on CPU, then subsequent steps use GPU encode to reduce readback.
- `dashiCORE/scripts/run_vulkan_core_mask_majority.py` — GPU smoke test for the `core_mask_majority` compute shader; validates Vulkan carrier dispatch vs CPU majority vote. Requires `VK_ICD_FILENAMES` and python-vulkan/glslc; accepts `--n` (elements per channel) and `--k` (channels).
 - Update policy: for fresh metrics, re-run the relevant script and append a timestamped summary to `COMPACTIFIED_CONTEXT.md`.

Recent GPU/vkFFT runs (user side):
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --backend vulkan --fft-backend vkfft-vulkan --dtype float64 --progress-every 5 --timing`
  - Result: completed; encode=17.69s, learn≈0, rollout≈0, decode_total=0.405s (0.068s/snap). vkFFT/Vulkan binding active (smoke: max error ~1.5e-6 on 256×256 complex64). ~600 MB system RAM; GPU VRAM not recorded.
- `... --N 1024 --steps 300 --stride 50 --backend vulkan --fft-backend vkfft-vulkan --dtype float64`
  - Result: encode overflow → NaNs; aborts at encode with `RuntimeError: Non-finite values in omega before encoding (try float64, smaller dt/Cs)`. Consider reducing `dt`/`Cs` or keeping N below ~1k for this LES integrator.
- Kernel-only reminder: provide a real z0 npz (`--z0-npz path`) or the runner will error (FileNotFound); kernel-only path skips LES entirely.
- Default params (N=64, steps=3000, stride=300, backend=vulkan, dtype=float64)
  - Result: saved `outputs/v4_t3000_compare.png`; timing encode=142.132s, learn=0.131s, rollout=0.235s, decode_total=0.043s (0.007s/snap).
- Example (streaming, saved traj, timing): `MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --save-traj outputs/traj_saved.npz --progress-every 5 --timing` → encode=7.967s, learn=0.221s, rollout=0.010s, decode_total=0.321s (0.054s/snapshot); files `outputs/v4_t0005_compare.png` … `v4_t0030_compare.png`.
- At very large grids (e.g., N=6400), float32 LES overflows; use `--dtype float64` or reduce `dt`/`Cs`, and a NaN guard will fail fast if ω contains non-finite values.
- Example (streaming, saved traj, timing): `MPLBACKEND=Agg python run_v4_snapshots.py --N 640 --steps 30 --stride 5 --out-dir outputs --save-traj outputs/traj_saved.npz --progress-every 5 --timing`  
  Output (on this machine): encode=7.967s, learn=0.221s, rollout=0.010s, decode_total=0.321s (0.054s/snapshot). Images saved: `outputs/v4_t0005_compare.png`, …, `outputs/v4_t0030_compare.png`.

## Follow-Ups
- Fix `naw.py` voxel grid shape (matplotlib voxels expects edge-aligned arrays).
- Install `scikit-image` or add a stub to unblock `naw2.py`.
- Save plots to files when running headless (Agg) instead of calling `plt.show()`.

## Other Utilities
- `make_kernel_artifacts.py` — generate kernel-only A/z artifacts from a v4 run.
- `scripts/plot_enstrophy.py` — plot enstrophy from JSON or CSV logs.
- `scripts/compare_les_gpu_cpu.py` — sanity check GPU LES vs CPU baseline.
- `scripts/validate_gpu_truth.py` — validate GPU LES against stored truth.
- `scripts/make_truth_3d.py` — pseudo-spectral 3D periodic incompressible
  truth generator for velocity/vorticity bridge artifacts; defaults to CPU and
  has an opt-in SPV/vkFFT GPU lane via `--backend gpu`.
- `scripts/ns_ev5_shell_enstrophy.py` — convert `make_truth.py` snapshots into
  shell-enstrophy and EV5 candidate diagnostic bundles.
- `scripts/run_sweep.py` / `scripts/perf_sampler.py` — parameter sweeps and perf sampling.
