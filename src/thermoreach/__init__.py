"""Thermochemical reachability study package.

Modules:
    toy:       isothermal A->B->C toy reactor, exact constant-control propagator,
               invariant barrier, analytical envelopes.
    controls:  admissible control-history classes (constant, pulse, bang-bang,
               randomized, structured, optimized).
    reactor:   detailed-chemistry adiabatic fixed-pressure CSTR RHS, integration,
               exact mixing balances, cross-checks against Cantera ReactorNet.
    envelopes: reference-set construction (steady family, constant-control
               transients, switching histories) and empirical envelope comparison.
    metrics:   dimensionless distance metrics and projections.
    splitting: Strang X_(dt/2) R_dt X_(dt/2) splitting of the CSTR ODE.
    io_utils:  run manifests, trajectory saving, hashing.
"""

__version__ = "0.1.0"
