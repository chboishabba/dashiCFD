# VFX free-surface producer boundary

The VFX fixture is a deterministic two-dimensional Saint-Venant/shallow-water solver using first-order Rusanov finite-volume fluxes and a fixed solid hull mask.

It may be used to produce bounded free-surface height, depth-averaged velocity, wake/disturbance fields, hydrostatic wall-load proxies, CFL/run metadata, and deterministic artifact receipts.

It is **not** a three-dimensional Volume-of-Fluid, RANS, LES, DNS, spray, cavitation, air-entrainment, viscous boundary-layer, fracture, moving-body, or naval-hydrodynamic validation model. A stable run is numerical evidence only; it is not physical calibration, structural FEA, or a Navier-Stokes regularity result.

The intended escalation path is:

```text
reduced shallow-water receipt
  -> same-shot geometry/timeline weld
  -> compare against a 3D free-surface producer on the same fixture
  -> retain discrepancy as a residual
  -> promote only coordinates paid by the higher-fidelity receipt
```

The existing periodic vorticity LES remains useful for turbulence/vorticity experiments but is not silently reinterpreted as this hull/free-surface problem.
