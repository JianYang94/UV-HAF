"""Reduced climate emulator + cheap multi-sphere diagnostics.

The climate core is a two-layer (surface + deep ocean) energy-balance model
(Geoffroy et al. 2013), the standard reduced form behind FaIR-class emulators:

    C_s dT/dt   = F(t) - lambda_fb * T - gamma * (T - T_d)
    C_d dT_d/dt = gamma * (T - T_d)

with feedback parameter lambda_fb = F2x / ECS. The free parameters calibrated by
the SMC are ECS (equilibrium climate sensitivity) and gamma (deep-ocean heat
uptake). C_s, C_d, F2x are fixed at AR6-consistent values.

CLIMATE CORE: when the `fair` package is importable, the two-layer EBM is
integrated by FaIR's `EnergyBalanceModel` (exact matrix-exponential solver +
validated TOA-imbalance/OHC bookkeeping) rather than the internal forward-Euler
loop. The two are physically identical (same C_s, C_d, same ECS/gamma meaning):
FaIR's emergent ECS = forcing_4co2 * 0.5 / ocean_heat_transfer[0], so setting
forcing_4co2 = 2*F2x and ocean_heat_transfer = [F2x/ECS, gamma] reproduces this
model exactly, only with the discretisation error removed. If `fair` is missing,
the code falls back to forward Euler.

OCEAN CHEMISTRY: surface pH and aragonite saturation are solved by PyCO2SYS (the
full nonlinear carbonate system, surface equilibrium with atmospheric CO2 at
fixed alkalinity) when the package is importable; a cheap linear form is used as
a fallback and inside the SMC (where pH/Omega do not enter the likelihood).

FURTHER PRODUCTION STEPS: a 3-layer FaIR EBM with AR6-calibrated thermal
structure, and the full emissions-driven FaIR carbon cycle.
"""
from __future__ import annotations

import numpy as np

from . import data

try:  # FaIR is the climate core when available; forward-Euler is the fallback
    from fair.energy_balance_model import EnergyBalanceModel as _FairEBM
    _HAS_FAIR = True
except Exception:  # pragma: no cover - exercised only without fair installed
    _HAS_FAIR = False

try:  # PyCO2SYS is the carbonate core when available; linear form is the fallback
    import PyCO2SYS as _pyco2
    _HAS_PYCO2SYS = True
except Exception:  # pragma: no cover - exercised only without PyCO2SYS installed
    _HAS_PYCO2SYS = False

# Fixed two-box parameters (Geoffroy et al. 2013, AR6-consistent)
C_S = 8.0      # W yr m-2 K-1, surface/upper-ocean heat capacity
C_D = 100.0    # W yr m-2 K-1, deep-ocean heat capacity
EARTH_AREA = 5.10e14   # m2
OCEAN_FRAC = 0.71
SECONDS_PER_YEAR = 3.1557e7

# Surface-ocean carbonate boundary conditions (PyCO2SYS)
OCEAN_TA = 2300.0      # umol/kg, total alkalinity (global surface mean)
OCEAN_SAL = 35.0       # PSU
OCEAN_T0 = 18.0        # degC, preindustrial global-mean SST (chemistry reference)

# default parameter vector and its names (what the SMC samples)
PARAM_NAMES = ("ecs", "gamma")
PRIOR = {
    "ecs":   ("normal", 3.0, 0.5),     # K, AR6-like prior
    "gamma": ("uniform", 0.3, 1.2),    # W m-2 K-1, deep-ocean heat uptake
}


def default_theta() -> dict:
    return {"ecs": 3.0, "gamma": 0.7}


def climate_core(years: np.ndarray | None = None) -> str:
    """Name of the climate integrator that will be used for `years`."""
    if _HAS_FAIR and (years is None or _uniform_annual(years)):
        return "FaIR EnergyBalanceModel (matrix-exponential, 2-layer Geoffroy)"
    return "in-house 2-box forward-Euler (FaIR unavailable)"


def chemistry_core(full: bool = True) -> str:
    """Name of the ocean-chemistry model that will be used."""
    if full and _HAS_PYCO2SYS:
        return "PyCO2SYS (full carbonate system, surface equilibrium)"
    return "linearised pH/Omega (fast fallback)"


def ocean_chemistry(co2, sst_anom, full: bool = True):
    """Surface-ocean pH (total scale) and aragonite saturation state Omega_arag.

    The surface mixed layer is taken to be in equilibrium with atmospheric CO2
    (par1 = pCO2 ~ atmospheric ppm) at fixed total alkalinity; SST = OCEAN_T0 +
    the model's SST anomaly, so FaIR-driven warming couples into the carbonate
    equilibria (warming raises Omega_arag at fixed pCO2, while rising CO2 lowers
    both pH and Omega -- the real nonlinear, buffered response).

    full=True uses PyCO2SYS; full=False (or PyCO2SYS missing) uses the cheap
    linearisation. pH/Omega do not enter the SMC likelihood, so the sampler runs
    with full=False to avoid ~50 ms/solve x thousands of particle evaluations.
    """
    co2 = np.asarray(co2, dtype=float)
    if full and _HAS_PYCO2SYS:
        r = _pyco2.sys(par1=co2, par1_type=4, par2=OCEAN_TA, par2_type=1,
                       temperature=OCEAN_T0 + np.asarray(sst_anom, dtype=float),
                       salinity=OCEAN_SAL, pressure=0.0)
        return (np.asarray(r["pH"], dtype=float),
                np.asarray(r["saturation_aragonite"], dtype=float))
    # linearised fallback (matches the proposal's linear d(pH)/dt form)
    ph = 8.2 - 0.0011 * (co2 - data.CO2_PI)
    omega = 3.4 - 0.0040 * (co2 - data.CO2_PI)
    return ph, omega


def _uniform_annual(years: np.ndarray) -> bool:
    """FaIR's EBM uses a single scalar timestep; only drive it on annual grids."""
    d = np.diff(np.asarray(years, dtype=float))
    return d.size > 0 and np.allclose(d, 1.0)


def _climate_fair(ecs, gamma, erf, n):
    """Geoffroy two-layer EBM via FaIR's exact matrix-exponential solver.

    Returns (T_surface, N) with N the TOA radiative imbalance (W m-2). ECS is set
    exactly through lambda = F2X/ECS (FaIR's emergent ECS = forcing_4co2*0.5 /
    ocean_heat_transfer[0]); gamma is the surface<->deep exchange, as in 2-box.
    """
    m = _FairEBM(
        ocean_heat_capacity=np.array([C_S, C_D]),
        ocean_heat_transfer=np.array([data.F2X / max(ecs, 0.3), gamma]),
        deep_ocean_efficacy=1.0,
        forcing_4co2=2.0 * data.F2X,
        n_timesteps=n,
    )
    m.add_forcing(np.asarray(erf, dtype=float), timestep=1.0)
    m.run()
    return m.temperature[:, 0], m.toa_imbalance


def _climate_twobox(ecs, gamma, erf, years):
    """In-house forward-Euler integration of the same 2-box EBM (FaIR fallback)."""
    lam = data.F2X / max(ecs, 0.3)
    n = len(years)
    T = np.zeros(n)
    Td = np.zeros(n)
    N = np.zeros(n)
    for i in range(1, n):
        dt = years[i] - years[i - 1]
        N[i - 1] = erf[i - 1] - lam * T[i - 1]
        T[i] = T[i - 1] + dt / C_S * (erf[i - 1] - lam * T[i - 1]
                                      - gamma * (T[i - 1] - Td[i - 1]))
        Td[i] = Td[i - 1] + dt / C_D * (gamma * (T[i - 1] - Td[i - 1]))
    N[-1] = erf[-1] - lam * T[-1]
    return T, N


def run_emulator(theta, years: np.ndarray, ssp: str = "ssp245",
                 chem: bool = True) -> dict:
    """Integrate the 2-box EBM over `years` and return the slice variables.

    Parameters
    ----------
    theta : dict or sequence aligned with PARAM_NAMES
    years : 1-D array of (annual) years, e.g. arange(1750, 2301)
    ssp   : scenario for the forcing extension after 2019

    Returns dict of annual trajectories: gmst, ohc, co2, ph, omega, sst (+year).
    All temperatures are anomalies on the model's own baseline; `chs`/`smc`
    re-baseline as needed.
    """
    if not isinstance(theta, dict):
        theta = dict(zip(PARAM_NAMES, theta))
    ecs = float(theta["ecs"])
    gamma = float(theta["gamma"])

    erf = data.total_erf(years, ssp=ssp)
    if _HAS_FAIR and _uniform_annual(years):
        T, N = _climate_fair(ecs, gamma, erf, len(years))
    else:
        T, N = _climate_twobox(ecs, gamma, erf, years)

    # Ocean heat content (ZJ): cumulative TOA imbalance over the ocean area.
    # (ocean-area convention matches the 0-2000 m global-ocean obs product used
    #  in the SMC; FaIR's own full-Earth-area OHC would over-count by ~1/0.71.)
    flux_to_J = N * SECONDS_PER_YEAR * EARTH_AREA * OCEAN_FRAC  # J yr-1
    ohc = np.cumsum(flux_to_J) / 1e21  # ZJ

    co2 = data.co2_pathway(years, ssp=ssp)
    sst = 0.87 * T  # SST anomaly tracks GMST with a land/ocean contrast factor
    # ocean carbonate system: PyCO2SYS when chem=True, linear form otherwise
    ph, omega = ocean_chemistry(co2, sst, full=chem)

    return {
        "year": years, "gmst": T, "ohc": ohc, "co2": co2,
        "ph": ph, "omega": omega, "sst": sst, "toa_imbalance": N,
    }


def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw `n` particles (shape [n, len(PARAM_NAMES)]) from the prior."""
    cols = []
    for name in PARAM_NAMES:
        kind, a, b = PRIOR[name]
        if kind == "normal":
            cols.append(rng.normal(a, b, n))
        elif kind == "uniform":
            cols.append(rng.uniform(a, b, n))
        else:
            raise ValueError(kind)
    return np.column_stack(cols)


def log_prior(theta_arr: np.ndarray) -> np.ndarray:
    """Log prior density for an array of particles [n, n_param]."""
    lp = np.zeros(len(theta_arr))
    for j, name in enumerate(PARAM_NAMES):
        kind, a, b = PRIOR[name]
        x = theta_arr[:, j]
        if kind == "normal":
            lp += -0.5 * ((x - a) / b) ** 2 - np.log(b * np.sqrt(2 * np.pi))
        elif kind == "uniform":
            lp += np.where((x >= a) & (x <= b), -np.log(b - a), -np.inf)
    return lp
