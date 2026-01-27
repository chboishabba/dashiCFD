# Signed Filament Annihilation

This note fixes the operator semantics used for coherence/annihilation and
keeps the interpretation separate from the operator.

## Signed filament field

At scale k, define a signed filament field:

F_k : Omega -> {-1, 0, +1}

with factorisation:

- support: sigma_k(x) in {0,1}
- sign: s_k(x) in {-1,+1}

and F_k(x) = sigma_k(x) * s_k(x).

The sign is defined relative to a fixed admissible orientation gauge.

## Coherence / entropy functional

Let C_k(x) be a scale-dependent coherence functional. Low C_k corresponds to
entropy-dominated incoherence; high C_k corresponds to persistent structure.

## PDA admissibility operator

Define the PDA admissibility operator A:

A(v) =
  +1  admissible (coherent, aligned)
   0  project (entropy-dominated)
  -1  reject (orientation violation)

## Annihilation / closure projector

Define the annihilation projector:

Pi_ann = A o A

Properties:

- Idempotent: Pi_ann(Pi_ann(x)) = Pi_ann(x)
- Non-expansive: never creates new support
- Information-preserving under admissible coarse-graining

## Annihilation rule (formal)

There exists a scale-dependent threshold eps_k such that:

F_{k+1}(x) =
  F_k(x)   if C_k(x) > eps_k
  0        if C_k(x) <= eps_k

Equivalently: F_{k+1} = Pi_ann(F_k).

Once annihilated, a filament does not reappear at coarser scales.

## Operational consequences

- Annihilation is a closure projector, not a modelling error.
- Coherent low-pass dynamics persist; incoherent residuals are projected to 0.
- Entropy saturation implies refinement adds no new information.
