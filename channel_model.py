"""
INTEGRATED 6G CHANNEL MODEL
Ray Tracer Ground Truth + Hybrid 3D GBSM + Statistical Validation
═══════════════════════════════════════════════════════════════════════
APMC 2026
═══════════════════════════════════════════════════════════════════════
DEPENDENCY ON RAY TRACER (TRANSPARENT INVENTORY)
─────────────────────────────────────────────────
The GBSM uses the following parameters fitted from RT data:
    Mixture weights:     w_los, w_sb_rx, w_sb_tx, w_db
    Scale parameters:    R1, R2, λ_excl  (= ln 2 · (R₂/r_50)² with
                                          r_50 the empirical half-
                                          exclusion radius)
    Concentrations:      κ_tx (Banerjee MLE), κ_int (Mardia MLE)
    Mode locations:      µ_hat, az_peaks, interior_phi_peaks
    (12 parameters total — comparable to standard 3GPP / METIS GBSMs)

The GBSM uses the following CALIBRATED constants where the value was
chosen with knowledge of RT output but is consistent with independent
physical bounds:
    ρ_b = 1.41    (within physical bracket [0.9, 1.8] from cabin clutter
                   density; calibrated such that SB-Rx peak ≈ −45°)

The GBSM and RT share the following inputs by definition:
    Materials:       INTERIOR_LOSS_PER_M, GLASS, METAL, SEAT, CONCRETE
    Bus geometry:    Tata Starbus Urban 12m specifications
    Deployment:      TX_POS, RX_POS, RX_VEL, FREQ
"""

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & SHARED CONSTANTS
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import (gaussian_kde, vonmises, truncnorm,
                         vonmises_fisher, ks_2samp, wasserstein_distance,
                         pearsonr)
from scipy.special import rel_entr
from scipy.integrate import trapezoid, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks

from dataclasses import dataclass
from typing import List, Optional
import os, warnings
warnings.filterwarnings("ignore")

# ── Physical constants ─────────────────────────────────────────────
C           = 3e8
FREQ        = 28e9
WAVELENGTH  = C / FREQ

# ── System budget ──────────────────────────────────────────────────
K_B               = 1.38e-23
T0                = 290.0
BW                = 100e6
NF_DB             = 7.0
NOISE_FLOOR_DBM   = 10*np.log10(K_B * T0 * BW * 1e3) + NF_DB
TX_POWER_DBM      = 43.0
THRESHOLD_DBM     = NOISE_FLOOR_DBM - 30.0
INTERIOR_LOSS_PER_M = 1.5  # dB/m

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — RAY TRACER (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Material:
    name: str
    reflectivity: float
    transmissivity: float
    reflection_loss_db: float
    transmission_loss_db: float

CONCRETE = Material("Concrete",       0.72, 0.00, 1.5, 100.0)
GLASS    = Material("Glass",          0.15, 0.78, 0.5,   3.5)
METAL    = Material("Metal",          0.95, 0.00, 0.3, 100.0)
SEAT     = Material("Seat/Passenger", 0.25, 0.00, 6.0, 100.0)

@dataclass
class RayPath:
    power_db:     float
    toa_ns:       float
    doppler_hz:   float
    aoa_elev_deg: float
    aoa_azim_deg: float
    path_type:    str
    num_bounces:  int
    path_points:  List[np.ndarray]

@dataclass
class BusSurface:
    points:   np.ndarray
    normal:   np.ndarray
    material: Material
    name:     str

class Bus:
    def __init__(self, center, length=12.0, width=2.5, height=3.5):
        self.center = np.array(center, dtype=float)
        self.L, self.W, self.H = length, width, height
        self.obstacles = []
        self.surfaces  = self._build_surfaces()

    def _build_surfaces(self):
        cx, cy, cz = self.center
        L, W, H    = self.L, self.W, self.H
        xn, xp     = cx - L/2, cx + L/2
        yn, yp     = cy - W/2, cy + W/2
        zn, zp     = cz,        cz + H
        zh         = cz + 1.0

        surfaces = []

        def make(p0, p1, p2, p3, normal, mat, name):
            surfaces.append(BusSurface(
                np.array([p0,p1,p2,p3], dtype=float),
                np.array(normal,        dtype=float),
                mat, name))

        make([xp,yn,zn],[xp,yp,zn],[xp,yp,zp],[xp,yn,zp],[1,0,0], GLASS,"Front Window")
        make([xn,yn,zn],[xn,yp,zn],[xn,yp,zp],[xn,yn,zp],[-1,0,0],GLASS,"Rear Window")
        make([xn,yn,zn],[xp,yn,zn],[xp,yn,zh],[xn,yn,zh],[0,-1,0], METAL,"Left Wall")
        make([xn,yp,zn],[xp,yp,zn],[xp,yp,zh],[xn,yp,zh],[0,1,0],  METAL,"Right Wall")
        make([xn,yn,zh],[xp,yn,zh],[xp,yn,zp],[xn,yn,zp],[0,-1,0], GLASS,"Left Window")
        make([xn,yp,zh],[xp,yp,zh],[xp,yp,zp],[xn,yp,zp],[0,1,0],  GLASS,"Right Window")
        make([xn,yn,zp],[xp,yn,zp],[xp,yp,zp],[xn,yp,zp],[0,0,1],  METAL,"Roof")
        make([xn,yn,zn],[xp,yn,zn],[xp,yp,zn],[xn,yp,zn],[0,0,-1], METAL,"Floor")

        num_rows, seat_len, row_pitch = 14, 0.5, 0.7
        start_x = xn + 1.0
        for i in range(num_rows):
            sx0, sx1 = start_x + i*row_pitch, start_x + i*row_pitch + seat_len
            sz0, sz1 = zn, zn + 1.1
            for sy0, sy1, bank in [(yn,-0.3,"L"), (0.3,yp,"R")]:
                self.obstacles.append((sx0,sx1,sy0,sy1,sz0,sz1))
                make([sx0,sy0,sz1],[sx1,sy0,sz1],[sx1,sy1,sz1],[sx0,sy1,sz1],[0,0,1], SEAT,f"{bank}-Seat-{i}-Top")
                make([sx1,sy0,sz0],[sx1,sy1,sz0],[sx1,sy1,sz1],[sx1,sy0,sz1],[1,0,0], SEAT,f"{bank}-Seat-{i}-Frt")
                make([sx0,sy0,sz0],[sx0,sy1,sz0],[sx0,sy1,sz1],[sx0,sy0,sz1],[-1,0,0],SEAT,f"{bank}-Seat-{i}-Bak")
                n_aisle = [0,1,0] if bank=="L" else [0,-1,0]
                sy_aisle = sy1 if bank=="L" else sy0
                make([sx0,sy_aisle,sz0],[sx1,sy_aisle,sz0],[sx1,sy_aisle,sz1],[sx0,sy_aisle,sz1],n_aisle,SEAT,f"{bank}-Seat-{i}-Aisle")
        return surfaces

    @staticmethod
    def quad_area(surface: BusSurface) -> float:
        p = surface.points
        return float(np.linalg.norm(np.cross(p[1]-p[0], p[3]-p[0])))


class RayTracer6G:
    def __init__(self, tx_pos, rx_pos, rx_velocity, buildings, bus: Bus):
        self.tx       = np.asarray(tx_pos,       dtype=float)
        self.rx       = np.asarray(rx_pos,       dtype=float)
        self.vel      = np.asarray(rx_velocity,  dtype=float)
        self.buildings = buildings
        self.bus      = bus
        self.paths: List[RayPath] = []

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v/n if n > 1e-9 else v

    def _friis(self, d: float) -> float:
        return (20*np.log10(max(d,0.1)) + 20*np.log10(FREQ)
                + 20*np.log10(4*np.pi/C))

    def _scatter_loss(self, d1, d2, area, mat, normal, p_src, p_hit, p_dst) -> float:
        vi = self._unit(p_hit-p_src); vs = self._unit(p_dst-p_hit)
        ci = abs(float(np.dot(vi, normal))); cs = abs(float(np.dot(vs, normal)))
        sigma = max(area * mat.reflectivity * ci * cs, 1e-9)
        return (30*np.log10(4*np.pi)
                + 20*np.log10(max(d1,0.1)) + 20*np.log10(max(d2,0.1))
                - 10*np.log10(sigma) - 20*np.log10(WAVELENGTH)
                + mat.reflection_loss_db)

    def _doppler_entry(self, entry_dir: np.ndarray) -> float:
        return -float(np.dot(self.vel, self._unit(entry_dir))) / WAVELENGTH

    def _clear(self, p1, p2) -> bool:
        delta = p2 - p1
        dist  = np.linalg.norm(delta)
        if dist < 1e-3: return True
        steps     = max(10, int(dist*5))
        step_delta = delta / steps
        pt         = p1.copy()
        obs_list   = self.bus.obstacles
        bldgs      = self.buildings
        # Bus interior bounding box derived from bus geometry (no hardcoded
        # values — uses actual bus center/extents).
        bx, by, bz = self.bus.center
        bL, bW = self.bus.L, self.bus.W
        ix_lo, ix_hi = bx - bL/2, bx + bL/2
        iy_lo, iy_hi = by - bW/2, by + bW/2
        for _ in range(1, steps):
            pt += step_delta
            if 0.0 <= pt[2] <= 1.1:
                if ix_lo <= pt[0] <= ix_hi and iy_lo <= pt[1] <= iy_hi:
                    for obs in obs_list:
                        if (obs[0]<=pt[0]<=obs[1] and obs[2]<=pt[1]<=obs[3]
                                and obs[4]<=pt[2]<=obs[5]):
                            return False
            elif pt[2] >= 0.0:
                if not (ix_lo<=pt[0]<=ix_hi and iy_lo<=pt[1]<=iy_hi):
                    for b in bldgs:
                        if b[0]<=pt[0]<=b[1] and b[2]<=pt[1]<=b[3] and pt[2]<=b[4]:
                            return False
        return True

    def _record(self, pts, losses, ptype, nbounce, entry_dir=None) -> bool:
        power = TX_POWER_DBM - sum(losses)
        if power < THRESHOLD_DBM: return False
        dist = sum(np.linalg.norm(pts[i+1]-pts[i]) for i in range(len(pts)-1))
        toa  = dist / C * 1e9
        last = self._unit(pts[-1]-pts[-2])
        elev = float(np.degrees(np.arcsin(np.clip(last[2],-1,1))))
        azim = float(np.degrees(np.arctan2(last[1], last[0])))
        fd   = self._doppler_entry(entry_dir if entry_dir is not None else last)
        self.paths.append(RayPath(power,toa,fd,elev,azim,ptype,nbounce,
                                  [p.copy() for p in pts]))
        return True

    @staticmethod
    def _sample_quad(surf: BusSurface):
        p = surf.points
        u,v = np.random.rand(), np.random.rand()
        return p[0] + u*(p[1]-p[0]) + v*(p[3]-p[0])

    def trace_window_direct(self, n_per_window=100):
        added = 0
        rsl   = 10*np.log10(max(n_per_window,1))
        for s in self.bus.surfaces:
            if s.material is not GLASS: continue
            for _ in range(n_per_window):
                wp = self._sample_quad(s)
                if not self._clear(self.tx,wp) or not self._clear(wp,self.rx): continue
                d1 = np.linalg.norm(wp-self.tx); d2 = np.linalg.norm(self.rx-wp)
                losses = [self._friis(d1), s.material.transmission_loss_db,
                          INTERIOR_LOSS_PER_M*d2, rsl]
                if self._record([self.tx,wp,self.rx], losses,'window_direct',1,
                                entry_dir=wp-self.tx): added+=1
        return added

    def trace_interior_bounce(self, n_win=30, n_int=5):
        added = 0
        rsl   = 10*np.log10(max(n_win,1))
        for win in self.bus.surfaces:
            if win.material is not GLASS: continue
            for _ in range(n_win):
                wp = self._sample_quad(win)
                if not self._clear(self.tx,wp): continue
                d1 = np.linalg.norm(wp-self.tx); entry = wp-self.tx
                for ref in self.bus.surfaces:
                    if ref.material is GLASS: continue
                    area_per = Bus.quad_area(ref) / max(n_int,1)
                    for _ in range(n_int):
                        rp = self._sample_quad(ref)
                        d2 = np.linalg.norm(rp-wp); d3 = np.linalg.norm(self.rx-rp)
                        if d2<0.05 or d3<0.05: continue
                        if not self._clear(wp,rp) or not self._clear(rp,self.rx): continue
                        vi=self._unit(rp-wp); vo=self._unit(self.rx-rp)
                        ci=abs(float(np.dot(vi,ref.normal))); co=abs(float(np.dot(vo,ref.normal)))
                        if ci<0.05 or co<0.05: continue
                        sigma = max(area_per*ref.material.reflectivity*ci*co, 1e-9)
                        il = (INTERIOR_LOSS_PER_M*(d2+d3) + ref.material.reflection_loss_db
                              - 10*np.log10(max(sigma/(4*np.pi),1e-12)))
                        losses=[self._friis(d1),win.material.transmission_loss_db,il,rsl]
                        if self._record([self.tx,wp,rp,self.rx],losses,'interior_bounce',2,
                                        entry_dir=entry): added+=1
        return added

    def trace_bldg_scatter(self, n_bldg=30, n_win=15):
        added = 0
        rsl   = 10*np.log10(max(n_win,1))
        for bldg in self.buildings:
            xn,xp,yn,yp,zh = bldg
            W_b,H_b  = xp-xn, zh
            cy_bldg  = (yn+yp)/2
            ny       = -1.0 if cy_bldg>0 else 1.0
            normal   = np.array([0.,ny,0.])
            area_per = W_b*H_b / max(n_bldg,1)
            for _ in range(n_bldg):
                bp = np.array([np.random.uniform(xn,xp), yn,
                               np.random.uniform(0.5,max(H_b-0.5,0.6))])
                if not self._clear(self.tx,bp): continue
                vi = self._unit(bp-self.tx)
                if abs(float(np.dot(vi,normal))) < 0.05: continue
                for win in self.bus.surfaces:
                    if win.material is not GLASS: continue
                    for _ in range(n_win):
                        wp = self._sample_quad(win)
                        if not self._clear(bp,wp) or not self._clear(wp,self.rx): continue
                        d1=np.linalg.norm(bp-self.tx)
                        d2=np.linalg.norm(wp-bp); d3=np.linalg.norm(self.rx-wp)
                        bl = self._scatter_loss(d1,d2,area_per,CONCRETE,normal,self.tx,bp,wp)
                        losses=[bl,win.material.transmission_loss_db,
                                INTERIOR_LOSS_PER_M*d3,rsl]
                        if self._record([self.tx,bp,wp,self.rx],losses,'bldg_scatter',2,
                                        entry_dir=wp-bp): added+=1
        return added

    def trace_bldg_interior(self, n_bldg=15, n_win=10, n_int=3):
        added = 0
        rsl   = 10*np.log10(max(n_win,1))
        for bldg in self.buildings:
            xn,xp,yn,yp,zh = bldg
            W_b,H_b = xp-xn, zh
            cy_bldg = (yn+yp)/2
            ny      = -1.0 if cy_bldg>0 else 1.0
            normal  = np.array([0.,ny,0.])
            ab      = W_b*H_b / max(n_bldg,1)
            for _ in range(n_bldg):
                bp = np.array([np.random.uniform(xn,xp), yn,
                               np.random.uniform(0.5,max(H_b-0.5,0.6))])
                if not self._clear(self.tx,bp): continue
                for win in self.bus.surfaces:
                    if win.material is not GLASS: continue
                    for _ in range(n_win):
                        wp = self._sample_quad(win)
                        if not self._clear(bp,wp): continue
                        d1=np.linalg.norm(bp-self.tx); d2=np.linalg.norm(wp-bp)
                        entry = wp-bp
                        for ref in self.bus.surfaces:
                            if ref.material is GLASS: continue
                            ar = Bus.quad_area(ref)/max(n_int,1)
                            for _ in range(n_int):
                                rp = self._sample_quad(ref)
                                d3=np.linalg.norm(rp-wp); d4=np.linalg.norm(self.rx-rp)
                                if d3<0.05 or d4<0.05: continue
                                if not self._clear(wp,rp) or not self._clear(rp,self.rx): continue
                                vi=self._unit(rp-wp); vo=self._unit(self.rx-rp)
                                ci=abs(float(np.dot(vi,ref.normal))); co=abs(float(np.dot(vo,ref.normal)))
                                if ci<0.05 or co<0.05: continue
                                sg=max(ar*ref.material.reflectivity*ci*co,1e-9)
                                il=(INTERIOR_LOSS_PER_M*(d3+d4)+ref.material.reflection_loss_db
                                    -10*np.log10(max(sg/(4*np.pi),1e-12)))
                                bl=self._scatter_loss(d1,d2,ab,CONCRETE,normal,self.tx,bp,wp)
                                losses=[bl,win.material.transmission_loss_db,il,rsl]
                                if self._record([self.tx,bp,wp,rp,self.rx],losses,
                                               'bldg_interior',3,entry_dir=entry): added+=1
        return added

    def run(self, label=""):
        self.paths = []
        print(f"\n{'═'*70}\n  {label}\n{'═'*70}")
        print(f"  Tx={self.tx}  Rx={self.rx}  |v|={np.linalg.norm(self.vel)*3.6:.1f} km/h")
        c1 = self.trace_window_direct(n_per_window=8000)
        print(f"  [1] Tx→Window→Rx                :  {c1}")
        c2 = self.trace_interior_bounce(n_win=30, n_int=5)
        print(f"  [2] Tx→Window→Interior→Rx       : {c2}")
        c3 = self.trace_bldg_scatter(n_bldg=30, n_win=15)
        print(f"  [3] Tx→Building→Window→Rx       : {c3}")
        c4 = self.trace_bldg_interior(n_bldg=15, n_win=10, n_int=3)
        print(f"  [4] Tx→Bldg→Win→Interior→Rx     :    {c4}")
        total = len(self.paths)
        print(f"  Total paths: {total}")
        if total:
            pwr = np.array([p.power_db  for p in self.paths])
            toa = np.array([p.toa_ns    for p in self.paths])
            fd  = np.array([p.doppler_hz for p in self.paths])
            print(f"  Power [dBm]: {pwr.max():.1f} / {pwr.mean():.1f} / {pwr.min():.1f}")
            print(f"  ToA   [ns] : {toa.min():.1f} – {toa.max():.1f}")
            print(f"  Doppler[Hz]: {fd.min():.0f} – {fd.max():.0f}")
        return self


def make_buildings(scenario: str, seed=42):
    rng = np.random.default_rng(seed)
    if scenario == 'dense':
        blds = []
        for xi in range(-60, 120, 28):
            if abs(xi-50) < 20: continue
            blds.append((float(xi),float(xi+20),-48.,-14.,rng.uniform(18,38)))
            blds.append((float(xi),float(xi+20), 14., 48.,rng.uniform(18,38)))
        return blds
    return [(-55.,-35.,-35.,-15.,20.),(80.,100.,-30.,-10.,17.),(-30.,-12.,18.,36.,22.)]


def smooth_pdf(data, weights=None, n=600, bw_value=None):

    if len(data) < 5: return None, None, None
    if weights is not None:
        wn = weights / weights.sum()
        idx = np.random.choice(len(data), size=min(10000,len(data)*10), p=wn, replace=True)
        sample_arr = data[idx]
    else:
        sample_arr = data
    if bw_value is None:
        kde = gaussian_kde(sample_arr, bw_method='scott')
        # gaussian_kde stores its bandwidth factor in .factor; the
        # actual bandwidth used for evaluation is .factor * std(data).
        bw_used = float(kde.factor) * float(np.std(sample_arr, ddof=1))
    else:
        bw_used = float(bw_value)
        std_data = float(np.std(sample_arr, ddof=1))
        if std_data <= 0:
            kde = gaussian_kde(sample_arr, bw_method='scott')
        else:
            kde = gaussian_kde(sample_arr, bw_method=bw_used / std_data)
    lo,hi = data.min(), data.max()
    m     = (hi-lo)*0.12 + 1e-6
    xs    = np.linspace(lo-m, hi+m, n)
    ys    = kde(xs)
    iv    = trapezoid(ys, xs)
    if iv > 1e-12: ys /= iv
    return xs, ys, bw_used


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — MLE PARAMETER EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _power_linear(paths):
    return np.array([10**(p.power_db/10) for p in paths]) if paths else np.array([0.])


def _hollow_ip_normalisation(lam: float) -> float:
    """
    I(λ) = ∫₀¹ u² (1 − u²) (1 − exp(−λ u²)) du
    Used as the λ-dependent part of the Hollow-IP normalisation constant.
    Computed by adaptive quadrature (1-D, fast).
    """
    val, _ = quad(lambda u: u*u * (1 - u*u) * (1 - np.exp(-lam * u*u)),
                  0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
    return max(val, 1e-12)


def fit_lambda_excl_mle(r_values: np.ndarray, weights: np.ndarray,
                        R2: float) -> float:
    """
    Density (over r ∈ [0, R₂]):
        f(r;λ) = [r²(1 − r²/R₂²)(1 − exp(−λ(r/R₂)²))] / [R₂³ · I(λ)]

    Log-likelihood (terms not in λ dropped):
        ℓ(λ) = Σ wᵢ log(1 − exp(−λ uᵢ²))  −  log I(λ)
        where uᵢ = rᵢ / R₂.

    The fitted λ has three equivalent representations, all reported in
    the parameter summary:
        λ_excl  : raw shape parameter ∈ [0.01, 500]
        φ_excl  : 1 − exp(−λ_excl) ∈ [0, 1) — dimensionless bounded form
        r_50    : R₂ · √(ln 2 / λ_excl) — half-exclusion radius in m
                  (the radius below which scatterer density is
                  suppressed by at least 50 %)
    """
    mask = (r_values > 1e-3) & (r_values < 0.99 * R2)
    if mask.sum() < 8:
        print("    [λ MLE] insufficient interior_bounce samples — fallback λ=5.0")
        return 5.0

    r = r_values[mask]
    w = weights[mask] / weights[mask].sum()
    u = r / R2
    u2 = u * u

    def neg_log_lik(lam):
        if lam <= 1e-4:
            return 1e12
        x = lam * u2
        log_surv = np.log1p(-np.exp(-x) + 1e-300)
        log_norm = np.log(_hollow_ip_normalisation(lam))
        return -(np.sum(w * log_surv)) + log_norm  # Σ w = 1

    # Widened bound: [0.01, 500]. v8 used [0.1, 50] which railed.
    res = minimize_scalar(neg_log_lik, bounds=(0.01, 500.0), method='bounded',
                          options={'xatol': 1e-2})
    if not res.success:
        print("    [λ MLE] optimiser failed — fallback λ=5.0")
        return 5.0

    # Sanity check: warn if we hit the bound (shouldn't with the new range,
    # but possible for pathological data).
    if res.x > 480.0:
        print(f"    [λ MLE] optimum near upper bound (λ={res.x:.1f}); "
              f"likelihood may be near-flat — verify with bound=2000")
    return float(res.x)


def detect_azimuth_peaks(paths_for_clustering: List[RayPath],
                         min_peak_separation_deg: float = 30.0,
                         max_peaks: int = 3) -> List[float]:


    if len(paths_for_clustering) < 20:
        return []

    az_deg = np.array([p.aoa_azim_deg for p in paths_for_clustering])
    az_rad = np.radians(az_deg)
    pw     = _power_linear(paths_for_clustering)
    pw     = pw / pw.sum()

    # ── Stage 1: histogram-based coarse peak detection ─────────────
    bins = np.linspace(-180, 180, 73)              # 5° bins
    hist, _ = np.histogram(az_deg, bins=bins, weights=pw)

    pad = np.concatenate([hist[-2:], hist, hist[:2]])
    sm  = np.convolve(pad, np.ones(5)/5.0, mode='valid')

    bin_w = 360 / (len(bins)-1)
    distance_bins = max(1, int(min_peak_separation_deg / bin_w))
    pk_idx, _ = find_peaks(sm, distance=distance_bins,
                           height=0.5*sm.max()/max_peaks)
    if len(pk_idx) == 0:
        return []

    heights = sm[pk_idx]
    order   = np.argsort(heights)[::-1][:max_peaks]
    centres = 0.5*(bins[:-1] + bins[1:])
    coarse_peaks_deg = centres[pk_idx[order]]

    # ── Stage 2: circular-mean refinement per cluster ──────────────
    # For each coarse peak, take samples within ±half-separation
    # and compute the weighted circular mean direction.
    half_window_rad = np.radians(min_peak_separation_deg / 2.0)
    refined_peaks_deg = []
    for cp_deg in coarse_peaks_deg:
        cp_rad = np.radians(cp_deg)
        # Circular distance to coarse peak
        dist = np.angle(np.exp(1j * (az_rad - cp_rad)))
        mask = np.abs(dist) <= half_window_rad
        if mask.sum() < 5:
            refined_peaks_deg.append(float(cp_deg))
            continue
        w_local = pw[mask]
        w_local = w_local / w_local.sum()
        cos_sum = np.sum(w_local * np.cos(az_rad[mask]))
        sin_sum = np.sum(w_local * np.sin(az_rad[mask]))
        refined_rad = np.arctan2(sin_sum, cos_sum)
        refined_peaks_deg.append(float(np.degrees(refined_rad)))

    return sorted(refined_peaks_deg)


def fit_vonmises_kappa_around_peaks(angles_rad: np.ndarray,
                                    weights: np.ndarray,
                                    peaks_rad: np.ndarray) -> float:

    if len(angles_rad) == 0 or len(peaks_rad) == 0:
        return 5.0
    w = weights / np.sum(weights)
    diffs = angles_rad[:, None] - peaks_rad[None, :]
    diffs = np.angle(np.exp(1j * diffs))                # wrap to (-π, π]
    nearest = np.argmin(np.abs(diffs), axis=1)
    res = diffs[np.arange(len(angles_rad)), nearest]
    cos_m = np.average(np.cos(res), weights=w)
    sin_m = np.average(np.sin(res), weights=w)
    R_bar = float(np.sqrt(cos_m**2 + sin_m**2))
    if R_bar < 0.53:
        kappa = R_bar * (2 - R_bar**2) / max(1 - R_bar**2, 1e-9)
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1 - R_bar, 1e-9)
    else:
        denom = R_bar**3 - 4 * R_bar**2 + 3 * R_bar
        kappa = 1.0 / max(denom, 1e-9)
    return float(np.clip(kappa, 0.5, 50.0))


def extract_params_from_rt_paths(paths: List[RayPath], tx_pos, rx_pos):
    """
    Power-weighted MLE extraction of GBSM parameters from RT path objects.
    Now also fits λ_excl and detects AoA azimuth peaks for mixture-vMF SB-Tx.
    """
    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)

    by_type = {t: [p for p in paths if p.path_type==t]
               for t in ('window_direct','interior_bounce',
                         'bldg_scatter','bldg_interior')}

    # ── Path-type mixture weights ───────────────────────────────────
    P = {k: _power_linear(v).sum() for k,v in by_type.items()}
    total_P = sum(P.values()) or 1.0
    w_los    = P['window_direct']  / total_P
    w_sb_rx  = P['interior_bounce'] / total_P
    w_sb_tx  = P['bldg_scatter']   / total_P
    w_db     = P['bldg_interior']  / total_P

    # ── R1: power-weighted Tx → bldg-scatter distance ──────────────
    bldg_paths = by_type['bldg_scatter']
    if bldg_paths:
        r1_vals = np.array([np.linalg.norm(p.path_points[1]-tx) for p in bldg_paths])
        r1_wts  = _power_linear(bldg_paths)
        R1 = float(np.average(r1_vals, weights=r1_wts/r1_wts.sum()))
        R1 = float(np.clip(R1, 10.0, 120.0))
    else:
        R1 = 35.0

    # ── κ (Banerjee MLE) and base mu_hat ───────────────────────────
    if len(bldg_paths) > 10:
        u_vecs, wts = [], []
        for p in bldg_paths:
            d = np.array(p.path_points[1]) - tx
            dn = np.linalg.norm(d)
            if dn < 0.1: continue
            u_vecs.append(d/dn)
            wts.append(10**(p.power_db/10))
        u_vecs = np.array(u_vecs); wts = np.array(wts)/np.sum(wts)
        mu_w   = np.average(u_vecs, axis=0, weights=wts)
        R_bar  = np.linalg.norm(mu_w)
        mu_hat = mu_w / (R_bar + 1e-12)

        def A_func(k):
            return (1.0/np.tanh(k)) - (1.0/k) - R_bar

        if   R_bar < 0.01:  kappa = 0.1
        elif R_bar > 0.999: kappa = 500.0
        else:               kappa = brentq(A_func, 1e-6, 500.0)
    else:
        kappa  = 5.0
        mu_hat = (rx-tx) / np.linalg.norm(rx-tx)

    # ── R2: 95th-percentile power-weighted Rx-side scatter radius ──
    int_paths = by_type['interior_bounce']
    if len(int_paths) > 5:
        r2_vals = np.array([np.linalg.norm(p.path_points[2]-rx) for p in int_paths])
        r2_wts  = _power_linear(int_paths)
        r2_wts /= r2_wts.sum()
        idx_srt = np.argsort(r2_vals)
        cum_w   = np.cumsum(r2_wts[idx_srt])
        R2 = float(r2_vals[idx_srt][np.searchsorted(cum_w, 0.95)])
        R2 = float(np.clip(R2, 1.0, 10.0))
    else:
        R2 = 3.5

    # ──  λ_excl MLE-fit from interior_bounce radii ──────────
    if len(int_paths) > 8:
        r_ib = np.array([np.linalg.norm(p.path_points[2]-rx) for p in int_paths])
        w_ib = _power_linear(int_paths)
        lambda_excl = fit_lambda_excl_mle(r_ib, w_ib, R2)
    else:
        lambda_excl = 5.0

    # ── Azimuth peaks for mixture vMF (from bldg_scatter AoA) ──
    az_peaks = detect_azimuth_peaks(bldg_paths)
    if not az_peaks:
        los_az_deg = float(np.degrees(np.arctan2((rx-tx)[1], (rx-tx)[0])))
        az_peaks = [los_az_deg]

    # ── Interior scatterer azimuth peaks ────────
    # SB-Rx and DB scatterers must reproduce the AoA azimuth lobes seen
    # in RT interior_bounce paths. Detect those AoA peaks, then convert
    # to SCATTERER azimuth peaks via
    #     phi_scatter = AoA_az + 180°  (mod 360°)
    # First pass: detect with provisional half_width = 25° to get κ_int
    PROVISIONAL_HALF_WIDTH_DEG = 25.0
    interior_aoa_peaks_deg_raw = detect_azimuth_peaks(
        int_paths, min_peak_separation_deg=40.0, max_peaks=4)

    los_az_deg = float(np.degrees(np.arctan2((rx-tx)[1], (rx-tx)[0])))

    def _circular_distance_deg(a, b):
        d = ((a - b + 180.0) % 360.0) - 180.0
        return abs(d)

    interior_aoa_peaks_deg_provisional = [
        p for p in interior_aoa_peaks_deg_raw
        if _circular_distance_deg(p, los_az_deg) > PROVISIONAL_HALF_WIDTH_DEG
    ]

    # Compute κ_int from provisional peak set
    if int_paths and interior_aoa_peaks_deg_provisional:
        int_az_rad   = np.radians(np.array([p.aoa_azim_deg for p in int_paths]))
        int_w        = _power_linear(int_paths)
        peaks_rad    = np.radians(np.array(interior_aoa_peaks_deg_provisional))
        kappa_int_provisional = fit_vonmises_kappa_around_peaks(
            int_az_rad, int_w, peaks_rad)
    else:
        kappa_int_provisional = 5.0

    # Second pass: re-filter using κ_int-derived half_width
    LOS_EXCLUSION_HALF_WIDTH_DEG = float(np.degrees(
        1.0 / np.sqrt(max(kappa_int_provisional, 0.5))))
    LOS_EXCLUSION_HALF_WIDTH_DEG = float(np.clip(
        LOS_EXCLUSION_HALF_WIDTH_DEG, 10.0, 60.0))

    interior_aoa_peaks_deg = [
        p for p in interior_aoa_peaks_deg_raw
        if _circular_distance_deg(p, los_az_deg) > LOS_EXCLUSION_HALF_WIDTH_DEG
    ]
    n_filtered = len(interior_aoa_peaks_deg_raw) - len(interior_aoa_peaks_deg)
    if interior_aoa_peaks_deg:
        interior_phi_peaks_deg = [
            ((p + 180.0 + 180.0) % 360.0) - 180.0   # wrap to (-180, 180]
            for p in interior_aoa_peaks_deg
        ]
    else:
        interior_phi_peaks_deg = []

    # Final κ_int: refit using the filtered peak set
    if int_paths and interior_aoa_peaks_deg:
        int_az_rad   = np.radians(np.array([p.aoa_azim_deg for p in int_paths]))
        int_w        = _power_linear(int_paths)
        peaks_rad    = np.radians(np.array(interior_aoa_peaks_deg))
        kappa_int    = fit_vonmises_kappa_around_peaks(
            int_az_rad, int_w, peaks_rad)
    else:
        kappa_int = 5.0

    # ── LoS temporal and Doppler statistics (DIAGNOSTIC ONLY in v3) ─
    # In v3, LoS ToA and Doppler are derived per-sample from the entry
    # point geometry, so these MLE-fitted Gaussian moments are kept only
    # for printing alongside the GBSM output for verification.
    wd_paths = by_type['window_direct']
    if len(wd_paths) > 3:
        toa_wd   = np.array([p.toa_ns     for p in wd_paths])
        dop_wd   = np.array([p.doppler_hz for p in wd_paths])
        pwr_wd   = _power_linear(wd_paths);  pwr_wd /= pwr_wd.sum()
        toa_los_mean_ns = float(np.average(toa_wd, weights=pwr_wd))
        toa_los_std_ns  = float(np.sqrt(
            np.average((toa_wd - toa_los_mean_ns)**2, weights=pwr_wd)))
        toa_los_std_ns  = max(toa_los_std_ns, 0.5)
        dop_los_mean    = float(np.average(dop_wd, weights=pwr_wd))
        dop_los_std     = float(np.sqrt(
            np.average((dop_wd - dop_los_mean)**2, weights=pwr_wd)))
        dop_los_std     = max(dop_los_std, 10.0)
    else:
        D_los           = np.linalg.norm(rx - tx)
        toa_los_mean_ns = D_los / C * 1e9
        toa_los_std_ns  = 2.0
        dop_los_mean    = -(FREQ / C) * 13.89 * 0.9
        dop_los_std     = 80.0

    phi_excl = 1.0 - np.exp(-lambda_excl)
    r_50_m   = R2 * np.sqrt(np.log(2) / max(lambda_excl, 1e-9))

    print(f"\n  ┌─ Extracted GBSM Parameters ──────────────────────────────")
    print(f"  │  w_LoS={w_los:.3f}  w_SB-Rx={w_sb_rx:.3f}  "
          f"w_SB-Tx={w_sb_tx:.3f}  w_DB={w_db:.3f}")
    print(f"  │  R1={R1:.2f}m  R2={R2:.2f}m  κ_tx={kappa:.2f}  "
          f"κ_int={kappa_int:.2f}")
    print(f"  │  λ_excl={lambda_excl:.2f}  (MLE)")
    print(f"  │    ↔ φ_excl=1-exp(-λ)={phi_excl:.6f}  (bounded form in [0,1))")
    print(f"  │    ↔ r_50={r_50_m*100:.1f} cm  (half-exclusion radius, R₂={R2:.2f}m)")
    print(f"  │  Bldg az peaks (deg)    : {[round(a,1) for a in az_peaks]}")
    print(f"  │  Interior AoA peaks raw : {[round(a,1) for a in interior_aoa_peaks_deg_raw]}")
    print(f"  │  LoS exclusion ±{LOS_EXCLUSION_HALF_WIDTH_DEG:.1f}° "
          f"(=1/√κ_int_provisional={kappa_int_provisional:.2f})")
    if n_filtered > 0:
        print(f"  │  Interior AoA filter    : removed {n_filtered} peak(s) "
              f"within ±{LOS_EXCLUSION_HALF_WIDTH_DEG:.1f}° of LoS az={los_az_deg:.1f}°")
    print(f"  │  Interior AoA peaks kept: {[round(a,1) for a in interior_aoa_peaks_deg]}")
    print(f"  │  Interior scat phi peaks: {[round(a,1) for a in interior_phi_peaks_deg]}")
    print(f"  │  LoS ToA (RT diag): mean={toa_los_mean_ns:.2f} ns  std={toa_los_std_ns:.2f} ns")
    print(f"  │  LoS Dop (RT diag): mean={dop_los_mean:.1f} Hz   std={dop_los_std:.1f} Hz")
    print(f"  └─────────────────────────────────────────────────────────")

    return dict(w_los=w_los, w_sb_rx=w_sb_rx, w_sb_tx=w_sb_tx, w_db=w_db,
                R1=R1, R2=R2, kappa=kappa, mu_hat=mu_hat, lambda_excl=lambda_excl,
                az_peaks=az_peaks,
                interior_phi_peaks_deg=interior_phi_peaks_deg,
                kappa_int=kappa_int,
                toa_los_mean_ns=toa_los_mean_ns, toa_los_std_ns=toa_los_std_ns,
                dop_los_mean=dop_los_mean, dop_los_std=dop_los_std)


def extract_rt_sample_arrays(paths: List[RayPath], n_samples=50_000):
    if not paths:
        empty = np.zeros(1)
        return {'toa': empty, 'el': empty, 'az': empty, 'doppler': empty}
    toa  = np.array([p.toa_ns       for p in paths])
    el   = np.array([p.aoa_elev_deg for p in paths])
    az   = np.array([p.aoa_azim_deg for p in paths])
    dop  = np.array([p.doppler_hz   for p in paths])
    pwr  = _power_linear(paths)
    wts  = pwr / pwr.sum()
    idx  = np.random.choice(len(paths), size=n_samples, p=wts, replace=True)
    return {'toa': toa[idx], 'el': el[idx], 'az': az[idx], 'doppler': dop[idx]}


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — STOCHASTIC APERTURE & BLOCKAGE
# ═══════════════════════════════════════════════════════════════════

def build_aperture_config(tx_pos, rx_pos, bus_geometry):
    """
    Angular aperture distribution W(θ,φ) from vehicle-class spec only.

    Each window face entry now also exposes its surface frame
    (centre, two tangent vectors, half-extents, outward normal) so that
    _sim_los can sample entry points uniformly across the face
    and compute per-sample Tx→entry direction Doppler.
    """
    bx, by, bz = bus_geometry['center']
    L, W, H    = bus_geometry['L'], bus_geometry['W'], bus_geometry['H']
    h_glass    = bus_geometry.get('h_glass', 1.0)
    H_glass    = H - h_glass
    rx         = np.asarray(rx_pos, dtype=float)
    tx         = np.asarray(tx_pos, dtype=float)

    z_ctr      = bz + h_glass + H_glass / 2.0

    # Window face geometry: (centre, normal, tangent_u, tangent_v, half_u, half_v)
    # normal points OUTWARD; tangents span the glass surface.
    windows = {
        'rear':  (np.array([bx-L/2, by, z_ctr]), np.array([-1.,0.,0.]),
                  np.array([0.,1.,0.]), np.array([0.,0.,1.]), W/2, H_glass/2),
        'front': (np.array([bx+L/2, by, z_ctr]), np.array([ 1.,0.,0.]),
                  np.array([0.,1.,0.]), np.array([0.,0.,1.]), W/2, H_glass/2),
        'left':  (np.array([bx, by-W/2, z_ctr]), np.array([0.,-1.,0.]),
                  np.array([1.,0.,0.]), np.array([0.,0.,1.]), L/2, H_glass/2),
        'right': (np.array([bx, by+W/2, z_ctr]), np.array([0., 1.,0.]),
                  np.array([1.,0.,0.]), np.array([0.,0.,1.]), L/2, H_glass/2),
    }

    # Equal alpha per window — shared RT/GBSM assumption.
    # Both pipelines assume per-face equal sampling weight under
    # far-field plane-wave illumination of a convex aperture set.
    #
    # Justification (shared by RT and GBSM):
    #   The Tx is at (0, 0, 30), the bus at (50, 0, 0..3.2). The Tx-to-
    #   bus distance is ~58 m, much larger than the bus's longest
    #   dimension (12 m), so the incident wavefront is approximately
    #   planar across all four window faces. A planar wave illuminates
    #   each face with the same intensity per unit area, so per-face
    #   contribution scales with face area only — and the per-sample
    #   Friis × interior_loss product handles the directional
    #   discrimination between windows close to vs far from Rx.
    #
    # RT implements this via n_per_window per face (uniform sampling on
    # each face independent of area). GBSM mirrors this by setting α
    # uniform across faces. The choice could equally have been
    # area-weighted in BOTH pipelines if a non-planar Tx pattern were
    # required — the assumption is symmetric, not a GBSM hack to match
    # RT.
    n_windows = len(windows)
    raw_alphas = {name: 1.0 for name in windows}
    total_raw = float(n_windows)

    config = []
    for name, (w_ctr, normal, t_u, t_v, hu, hv) in windows.items():
        alpha = raw_alphas[name] / total_raw

        to_rx    = rx - w_ctr
        d_rx     = np.linalg.norm(to_rx)
        u_to_rx  = to_rx / (d_rx + 1e-12)
        mu_theta = float(np.arcsin(np.clip(u_to_rx[2], -1, 1)))
        mu_phi   = float(np.arctan2(u_to_rx[1], u_to_rx[0]))

        # Half-aperture angular spreads from window surface dimensions
        sigma_theta = max(float(np.arctan2(hv, d_rx)), np.radians(3))
        kappa_phi   = max(1.0 / max(float(np.arctan2(hu, d_rx))**2, 1e-4), 1.0)
        kappa_phi   = min(kappa_phi, 30.0)

        d_tx_w = float(np.linalg.norm(w_ctr - tx))

        config.append({
            'name':        name,
            'alpha':       alpha,
            'mu_theta':    mu_theta,
            'sigma_theta': sigma_theta,
            'mu_phi':      mu_phi,
            'kappa_phi':   kappa_phi,
            'center_3d':   w_ctr,
            'normal':      normal,
            'tangent_u':   t_u,
            'tangent_v':   t_v,
            'half_u':      hu,
            'half_v':      hv,
            'd_rx':        d_rx,
            'd_tx_w':      d_tx_w,
        })
    return config


def compute_aperture_weight(theta_rad, phi_rad, aperture_config):
    """
    Analytical W(θ,φ) — used for paper figures only.
    Not used as a sample importance weight (samples are already drawn
    from this distribution; multiplying by it again would double-count).
    """
    W = np.zeros_like(theta_rad, dtype=float)
    for ap in aperture_config:
        # Truncnorm bounds: physical clamp to [-π/2, π/2].
        # The asymmetry around mu_theta is correct — it is the truncation
        # of an unbounded normal to the physical elevation range.
        # In practice (sigma_theta ≈ 0.18 rad, mu_theta ≈ ±0.15 rad),
        # the bounds are ≥6σ wide, so no meaningful clipping occurs.
        a_c = (-np.pi/2 - ap['mu_theta']) / ap['sigma_theta']
        b_c = ( np.pi/2 - ap['mu_theta']) / ap['sigma_theta']
        W  += (ap['alpha']
               * truncnorm.pdf(theta_rad, a_c, b_c,
                               loc=ap['mu_theta'], scale=ap['sigma_theta'])
               * vonmises.pdf(phi_rad, ap['kappa_phi'], loc=ap['mu_phi']))
    return np.maximum(W, 0)


def stochastic_blockage_probability(theta_rad, phi_rad,
                                    rho_b=1.41, A_b=1.0, h_vehicle=3.5):
    """
    Continuous in-bus clutter blockage model.

    Functional form:
        P_clutter(θ) = exp(−ρ_b · A_b · |cos θ|)   for θ ∈ (−85°, +15°)
        P_clutter(θ) = exp(−50)  ≈ 0               outside (chassis/roof)

    Path-length convention:
      |cos θ| is the dimensionless EFFECTIVE path through the cabin
      seat-clutter zone, normalised: maximum (=1) at horizontal arrival
      (θ = 0°, ray traverses the full cabin along its longest dimension)
      and zero at vertical (θ = ±90°, no clutter).

    Boundary cutoffs (derived from chassis geometry):
      • Roof at +15°: signals arriving from above the horizon at a
        sharper angle than +15° would have to penetrate the metal roof
        before reaching Rx. tan(15°) ≈ 0.27 corresponds to a ray
        entering through the upper edge of glass (z = 3.2 m) and
        reaching Rx (z = 1.2 m) along a 7.5 m horizontal cabin run.

      • Floor at −85° derived from chassis geometry, NOT RT data.
        The chassis floor is at z = 0; Rx is at z = 1.2 m. The steepest
        physically plausible scatterer-to-Rx arrival is from a scatterer
        at floor level separated horizontally from Rx by d_min = 0.1 m
        (closest plausible cabin-clutter position). Such a scatterer has
        elevation θ = −arctan(1.2 / 0.1) = −85.2°. Steeper arrivals
        would require scatterers within 10 cm of Rx — not a physically
        reasonable cabin-clutter geometry.

    Coefficient ρ_b: HONEST CALIBRATION DISCLOSURE.
      ρ_b is calibrated such that the post-blockage SB-Rx peak elevation
      sits near θ_peak = −45°. Solving d/dθ[cos θ · exp(−ρ_b A_b cos θ)] = 0
      for θ ∈ (−π/2, 0) gives cos(θ_peak) = 1/(ρ_b A_b), so the choice
      θ_peak = −45° determines ρ_b A_b = √2 ≈ 1.414.

      The chosen ρ_b A_b = 1.41 is consistent with an INDEPENDENT
      physical bracket from cabin clutter density:
          • 14 rows × 4 seats = 56 absorbers in the seat zone
          • Seat-zone volume: L · W · h_seat = 12 × 2.58 × 1.1 ≈ 34 m³
          • Volume density: ρ_vol ≈ 1.65 m⁻³
          • Effective absorber cross-section per seat at 28 GHz: 0.5 m²
            (with high seat-back transmission loss, ~6 dB per traversal)
          • Beer-Lambert linear coefficient from 3D density:
              ρ_b A_b ≈ ρ_vol · σ_seat · h_seat ≈ 1.65 · 0.5 · 1.1 ≈ 0.91
            with the upper bound from σ_seat = 1.0 m² giving 1.82.
          • Calibrated value 1.41 falls within physical bracket [0.9, 1.8].

      The specific value 1.41 was chosen with knowledge of the RT
      empirical median (−46°) rather than predicted from clutter density
      alone. The bracket argument shows the calibrated value is
      physically reasonable; it does not eliminate the calibration step.
    """
    theta_rad = np.atleast_1d(np.array(theta_rad, dtype=float))
    phi_rad   = np.atleast_1d(np.array(phi_rad,   dtype=float))

    # Soft sigmoid roll-offs at roof (+15°) and floor (−85°).
    # Replaces v8's hard P=exp(-50) cutoffs. The hard cutoffs created
    # support mismatch with RT — RT has small but non-zero density
    # at the tails (e.g., θ ≈ −90° from steep ceiling bounces), while
    # v8 had a hard zero. This caused asymmetric KL (0.26) >> JS (0.016)
    # on elevation, the tell-tale signature of support mismatch.
    #
    # Sigmoid width 5° on each side: smooth enough to eliminate the
    # hard-zero region, narrow enough to preserve the physical roof/floor
    # boundaries.
    SIGMOID_WIDTH_RAD = np.radians(5.0)
    roof_edge  = np.radians(15.0)
    floor_edge = np.radians(-85.0)

    # Roof sigmoid: 1 below roof_edge, drops smoothly to 0 above
    p_roof  = 1.0 / (1.0 + np.exp((theta_rad - roof_edge) / SIGMOID_WIDTH_RAD))
    # Floor sigmoid: 1 above floor_edge, drops smoothly to 0 below
    p_floor = 1.0 / (1.0 + np.exp((floor_edge - theta_rad) / SIGMOID_WIDTH_RAD))

    # Continuous clutter attenuation (unchanged from v8).
    L_eff       = np.abs(np.cos(theta_rad))
    p_clutter   = np.exp(-rho_b * A_b * L_eff)

    P = p_roof * p_floor * p_clutter
    return np.clip(P, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — HYBRID GBSM
# ═══════════════════════════════════════════════════════════════════

class HybridGBSM_Rigorous:

    def __init__(self, tx_pos, rx_pos, bus_geometry,
                 R1, R2, lambda_excl, kappa, mu_hat,
                 w_los, w_sb_rx, w_sb_tx, w_db,
                 use_mixture_vmf=True, street_azimuth_peaks=None,
                 interior_phi_peaks_deg=None, kappa_int=5.0,
                 interior_dB_per_m=1.5, glass_dB=3.5):

        self.tx    = np.asarray(tx_pos, dtype=float)
        self.rx    = np.asarray(rx_pos, dtype=float)
        self.v_vec = np.array([13.89, 0.0, 0.0])

        self.los_vec = self.rx - self.tx
        self.D       = float(np.linalg.norm(self.los_vec))

        self.R1          = float(R1)
        self.R2          = float(R2)
        self.lambda_excl = float(lambda_excl)
        self.kappa       = float(max(kappa, 0.1))
        self.fc          = FREQ

        # Interior attenuation coefficient and glass loss
        self.interior_dB_per_m = float(interior_dB_per_m)
        self.glass_dB          = float(glass_dB)

        tot = w_los + w_sb_rx + w_sb_tx + w_db or 1.0
        self.w_los   = w_los   / tot
        self.w_sb_rx = w_sb_rx / tot
        self.w_sb_tx = w_sb_tx / tot
        self.w_db    = w_db    / tot

        self.aperture_config = build_aperture_config(tx_pos, rx_pos, bus_geometry)

        # ── vMF mean directions (mixture for SB-Tx) ─────────────────
        self.mu_hat = np.asarray(mu_hat, dtype=float)
        self.mu_hat /= np.linalg.norm(self.mu_hat) + 1e-12

        self.use_mixture_vmf = use_mixture_vmf
        if street_azimuth_peaks and len(street_azimuth_peaks) > 1 and use_mixture_vmf:
            elev = float(np.arcsin(np.clip(self.mu_hat[2], -1, 1)))
            self.mu_vecs = [np.array([np.cos(elev)*np.cos(np.radians(a)),
                                      np.cos(elev)*np.sin(np.radians(a)),
                                      np.sin(elev)])
                            for a in street_azimuth_peaks]
        else:
            self.mu_vecs = [self.mu_hat]

        # Interior scatterer phi mixture (for SB-Rx and DB)
        if interior_phi_peaks_deg:
            self.interior_phi_peaks_rad = np.radians(np.array(interior_phi_peaks_deg))
        else:
            self.interior_phi_peaks_rad = None
        self.kappa_int = float(np.clip(kappa_int, 0.5, 50.0))

    # ── Hollow-IP rejection sampler — bounded with fallback ─
    def _sample_hollow_ip_r(self, N: int, max_iters: int = 60) -> np.ndarray:
        """
        Samples from f(r) ∝ r²(1 − r²/R₂²) · (1 − exp(−λ(r/R₂)²)) on [0, R₂].

        Bounded rejection: explicit max_iters cap. If insufficient samples
        accumulate, falls back to inverse-CDF sampling on a fine grid.
        """
        accepted = []
        # Analytical supremum of the unnormalised weight over r ∈ [0, R₂]:
        # max of u²(1-u²)(1 - exp(-λu²)). Compute once over a grid.
        u_grid  = np.linspace(0, 1, 500)
        w_grid  = u_grid**2 * (1 - u_grid**2) * (1 - np.exp(-self.lambda_excl*u_grid**2))
        sup_w   = w_grid.max() * 1.05  # 5% headroom

        batch = max(int(N * 4), 1024)
        for it in range(max_iters):
            r   = np.random.uniform(0, self.R2, batch)
            u   = r / self.R2
            w   = u**2 * np.maximum(1 - u**2, 0) * (1 - np.exp(-self.lambda_excl*u**2))
            acc = np.random.uniform(0, sup_w, batch) < w
            accepted.extend(r[acc].tolist())
            if len(accepted) >= N:
                return np.array(accepted[:N])

        # Fallback: inverse-CDF on the analytical density
        print(f"    [hollow-IP] rejection cap hit at it={max_iters}; "
              f"using inverse-CDF fallback ({len(accepted)}/{N} accepted).")
        cdf = np.cumsum(w_grid)
        cdf = cdf / cdf[-1]
        u01 = np.random.uniform(0, 1, N - len(accepted))
        idx = np.searchsorted(cdf, u01)
        idx = np.clip(idx, 0, len(u_grid)-1)
        accepted.extend((self.R2 * u_grid[idx]).tolist())
        return np.array(accepted[:N])

    def _path_weight(self, d_total_m, interior_dB=0.0, glass_dB=None):
        """
        Path weight with FSPL × glass × interior attenuation.

        Parameters
        ----------
        d_total_m   : total path length (m), used for free-space path loss
        interior_dB : interior attenuation in dB (1.5 dB/m × interior segment).
                      Per-sample for LoS/SB-Rx/DB; fixed average for SB-Tx.
        glass_dB    : glass penetration loss in dB. Defaults to self.glass_dB.
        """
        if glass_dB is None:
            glass_dB = self.glass_dB
        lam  = C / self.fc
        fspl = (lam / (4*np.pi*np.maximum(d_total_m, 0.1)))**2
        gl   = 10.0**(-glass_dB / 10.0)
        il   = 10.0**(-interior_dB / 10.0)
        return fspl * gl * il

    def _doppler(self, unit_entry):
        if unit_entry.ndim == 1:
            return -(self.fc/C) * unit_entry.dot(self.v_vec)
        return -(self.fc/C) * unit_entry @ self.v_vec

    def _sample_vmf(self, N):
        if self.use_mixture_vmf and len(self.mu_vecs) > 1:
            n_each = N // len(self.mu_vecs)
            parts  = [vonmises_fisher.rvs(mu, kappa=self.kappa, size=n_each)
                      for mu in self.mu_vecs]
            rem    = N - n_each*len(self.mu_vecs)
            if rem > 0:
                parts.append(vonmises_fisher.rvs(self.mu_vecs[0], kappa=self.kappa,
                                                  size=rem))
            return np.vstack(parts)
        return vonmises_fisher.rvs(self.mu_vecs[0], kappa=self.kappa, size=N)

    def _sample_interior_phi(self, N):
        """
        Sample scatterer azimuth phi2 from a vMF mixture biased
        toward the detected interior AoA peaks (rotated by 180°).
        Falls back to uniform [-π, π] when no peaks were detected.
        """
        if self.interior_phi_peaks_rad is None or len(self.interior_phi_peaks_rad) == 0:
            return np.random.uniform(-np.pi, np.pi, N)
        K = len(self.interior_phi_peaks_rad)
        n_each = N // K
        parts = [vonmises.rvs(self.kappa_int, loc=mu, size=n_each)
                 for mu in self.interior_phi_peaks_rad]
        rem = N - n_each * K
        if rem > 0:
            parts.append(vonmises.rvs(self.kappa_int,
                                      loc=self.interior_phi_peaks_rad[0], size=rem))
        return np.concatenate(parts)

    # ── LoS — fully geometric per-sample ────────────────────
    def _sim_los(self, N):
        """
        For each aperture face:
          1. Sample n_ap entry points uniformly across the window face
          2. Derive ToA, AoA elevation, AoA azimuth, and Doppler PER SAMPLE
             from that entry point (all consistent geometry)
          3. Weight by FSPL × glass × interior(d_rx) where interior loss
             is 1.5 dB/m of the in-bus segment from entry to Rx

        No MLE Gaussian for ToA or Doppler — those distributions emerge
        directly from window-surface sampling and entry geometry.
        """
        toa_l, el_l, az_l, dop_l, wt_l = [], [], [], [], []
        for ap in self.aperture_config:
            n_ap = max(int(N * ap['alpha']), 1)

            # 1. Surface entry points: ep = ctr + U·tu·hu + V·tv·hv
            U_face = np.random.uniform(-1, 1, n_ap)
            V_face = np.random.uniform(-1, 1, n_ap)
            entry_pts = (ap['center_3d'][None,:]
                         + U_face[:,None] * ap['tangent_u'][None,:] * ap['half_u']
                         + V_face[:,None] * ap['tangent_v'][None,:] * ap['half_v'])

            # 2a. Distances
            d_tx_ep = np.linalg.norm(entry_pts - self.tx[None,:], axis=1)
            d_ep_rx = np.linalg.norm(self.rx[None,:] - entry_pts, axis=1)

            # ToA: purely geometric per-sample. Each entry-point yields its
            # own (Tx → entry → Rx) path length. No Gaussian jitter — the
            # underlying distribution is fully determined by the surface
            # sampling. (v5–v7 added a 1 ns Gaussian for KDE-bandwidth
            # parity with RT's Scott's-rule estimator. v8 removes it: the
            # model output is the asymptotic distribution, the comparison
            # to RT carries an irreducible estimator-variance gap that is
            # acknowledged in the validation section, not papered over.)
            toa_s = (d_tx_ep + d_ep_rx) / C

            # 2b. AoA at Rx — direction (Rx − entry) / ||·||
            u_aoa  = (self.rx[None,:] - entry_pts) / (d_ep_rx[:,None] + 1e-12)
            aoa_el = np.degrees(np.arcsin(np.clip(u_aoa[:,2], -1, 1)))
            aoa_az = np.degrees(np.arctan2(u_aoa[:,1], u_aoa[:,0]))

            # 2c. Doppler from Tx→entry direction (entry direction of signal
            #     into the vehicle — physically correct for window arrival)
            u_entry = (entry_pts - self.tx[None,:]) / (d_tx_ep[:,None] + 1e-12)
            dop_s   = -(self.fc / C) * (u_entry @ self.v_vec)

            # 3. Path weight: Friis(d_tx) × glass × interior(d_rx)
            #    Matches RT's loss decomposition exactly. Previous v3 used
            #    FSPL on (d_tx + d_rx), which is a different (and biased)
            #    distance for the inverse-square law.
            lam_c       = C / self.fc
            fspl_d1     = (lam_c / (4 * np.pi * np.maximum(d_tx_ep, 0.1)))**2
            glass_lin   = 10.0 ** (-self.glass_dB / 10.0)
            interior_dB = self.interior_dB_per_m * d_ep_rx
            interior_lin = 10.0 ** (-interior_dB / 10.0)
            wt_s        = fspl_d1 * glass_lin * interior_lin

            toa_l.append(toa_s); el_l.append(aoa_el); az_l.append(aoa_az)
            dop_l.append(dop_s); wt_l.append(wt_s)

        def cat(lst): return np.concatenate(lst)[:N]
        return cat(toa_l), cat(el_l), cat(az_l), cat(dop_l), cat(wt_l)

    def _sim_sb_rx(self, N):
        """
        SB-Rx: scatterers on Hollow-IP ball around Rx.
        phi2 drawn from vMF mixture seeded by RT interior AoA peaks.
        interior loss = 1.5 dB/m × r2 (scatterer-to-Rx interior path).
        """
        r2         = self._sample_hollow_ip_r(N)
        cos_th     = np.random.uniform(-1, 1, N)
        phi2       = self._sample_interior_phi(N)            # ← biased
        sin_th     = np.sqrt(1 - cos_th**2)
        s2_local   = np.column_stack((r2*sin_th*np.cos(phi2),
                                      r2*sin_th*np.sin(phi2),
                                      r2*cos_th))
        s2_global  = self.rx + s2_local
        d_tx_s2    = np.linalg.norm(s2_global - self.tx, axis=1)
        toa        = (d_tx_s2 + r2) / C

        u_aoa      = (self.rx - s2_global) / r2[:,None]
        aoa_el     = np.degrees(np.arcsin(np.clip(u_aoa[:,2], -1, 1)))
        aoa_az     = np.degrees(np.arctan2(u_aoa[:,1], u_aoa[:,0]))

        p_block    = stochastic_blockage_probability(np.radians(aoa_el),
                                                     np.radians(aoa_az))
        interior_dB = self.interior_dB_per_m * r2
        wt          = self._path_weight(toa * C,
                                        interior_dB=interior_dB) * p_block

        u_entry    = (s2_global - self.tx) / d_tx_s2[:,None]
        dop        = self._doppler(u_entry)
        return toa, aoa_el, aoa_az, dop, wt

    def _sim_sb_tx(self, N):
        """
        SB-Tx: scatterer on vMF sphere around Tx.
        Adds a fixed-average interior segment (1.5 m) for the
        post-glass portion of the path inside the bus.
        """
        s1_dirs   = self._sample_vmf(N)
        s1_global = self.tx + self.R1 * s1_dirs
        d_s1_rx   = np.linalg.norm(self.rx - s1_global, axis=1)
        toa       = (self.R1 + d_s1_rx) / C

        u_aoa     = (self.rx - s1_global) / d_s1_rx[:,None]
        aoa_el    = np.degrees(np.arcsin(np.clip(u_aoa[:,2], -1, 1)))
        aoa_az    = np.degrees(np.arctan2(u_aoa[:,1], u_aoa[:,0]))

        p_block   = stochastic_blockage_probability(np.radians(aoa_el),
                                                    np.radians(aoa_az))
        # Interior segment after the window (~half-cabin-width average)
        interior_dB = self.interior_dB_per_m * 1.5
        wt        = self._path_weight(toa * C,
                                      interior_dB=interior_dB) * p_block
        dop       = self._doppler(u_aoa)
        return toa, aoa_el, aoa_az, dop, wt

    def _sim_db(self, N):
        """
        DB: Tx→bldg-scatter (vMF) → window → seat-scatter (Hollow-IP) → Rx.
        phi2 drawn from interior peak-biased vMF mixture.
        Interior segment = r2 + 1.0 m (window-cross to s2 + s2 to Rx).
        """
        s1_dirs   = self._sample_vmf(N)
        s1_global = self.tx + self.R1 * s1_dirs

        r2         = self._sample_hollow_ip_r(N)
        cos_th     = np.random.uniform(-1, 1, N)
        phi2       = self._sample_interior_phi(N)           # ← biased
        sin_th     = np.sqrt(1 - cos_th**2)
        s2_global  = self.rx + np.column_stack(
            (r2*sin_th*np.cos(phi2), r2*sin_th*np.sin(phi2), r2*cos_th))

        d_s1_s2   = np.linalg.norm(s2_global - s1_global, axis=1)
        toa       = (self.R1 + d_s1_s2 + r2) / C

        u_aoa     = (self.rx - s2_global) / r2[:,None]
        aoa_el    = np.degrees(np.arcsin(np.clip(u_aoa[:,2], -1, 1)))
        aoa_az    = np.degrees(np.arctan2(u_aoa[:,1], u_aoa[:,0]))

        p_block   = stochastic_blockage_probability(np.radians(aoa_el),
                                                    np.radians(aoa_az))
        interior_dB = self.interior_dB_per_m * (r2 + 1.0)
        wt        = self._path_weight(toa * C,
                                      interior_dB=interior_dB) * p_block

        u_entry   = (s2_global - s1_global) / d_s1_s2[:,None]
        dop       = self._doppler(u_entry)
        return toa, aoa_el, aoa_az, dop, wt

    def simulate_all_paths(self, num_samples=500_000):
        N = num_samples
        print(f"  Simulating {N:,} samples per path type …")
        los_out   = self._sim_los(N)
        sb_rx_out = self._sim_sb_rx(N)
        sb_tx_out = self._sim_sb_tx(N)
        db_out    = self._sim_db(N)
        keys = ('toa','el','az','doppler','weight')
        out  = {name: dict(zip(keys, arrays))
                for name, arrays in (('los',  los_out),   ('sb_rx', sb_rx_out),
                                     ('sb_tx', sb_tx_out),('db',    db_out))}
        print("  Done.")
        return out


def aggregate_gbsm_samples(sim_data, model_weights, n_samples=50_000):
    counts = {'los':   int(model_weights['w_los']   * n_samples),
              'sb_rx': int(model_weights['w_sb_rx'] * n_samples),
              'sb_tx': int(model_weights['w_sb_tx'] * n_samples),
              'db':    int(model_weights['w_db']    * n_samples)}
    out = {k: [] for k in ('toa','el','az','doppler')}
    for ptype, cnt in counts.items():
        if cnt == 0: continue
        pd = sim_data[ptype]
        wt = pd['weight']
        vm = wt > 0
        if not np.any(vm): continue
        vw = wt[vm]; vw /= vw.sum()
        idx = np.random.choice(np.where(vm)[0], size=cnt, p=vw, replace=True)
        out['toa'].extend(pd['toa'][idx] * 1e9)
        out['el'] .extend(pd['el'] [idx])
        out['az'] .extend(pd['az'] [idx])
        out['doppler'].extend(pd['doppler'][idx])
    return {k: np.array(v) for k,v in out.items()}


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — VALIDATION (KS p-value removed; W1/range added)
# ═══════════════════════════════════════════════════════════════════

def compute_validation_metrics(rt_samples, gbsm_samples,
                               n_bins=50, variable_name="ToA")
    rt   = np.asarray(rt_samples).flatten()
    gbsm = np.asarray(gbsm_samples).flatten()
    if len(rt) < 2 or len(gbsm) < 2:
        print(f"  [SKIP] {variable_name}: insufficient samples.")
        return {}

    xlo, xhi  = min(rt.min(), gbsm.min()), max(rt.max(), gbsm.max())
    var_range = max(xhi - xlo, 1e-12)
    edges     = np.linspace(xlo, xhi, n_bins+1)
    eps       = 1e-10

    p_rt, _ = np.histogram(rt,   bins=edges, density=True)
    p_gb, _ = np.histogram(gbsm, bins=edges, density=True)
    p_rt    = p_rt + eps;  p_rt /= p_rt.sum()
    p_gb    = p_gb + eps;  p_gb /= p_gb.sum()

    kl_div = float(np.sum(rel_entr(p_rt, p_gb)))
    m      = 0.5*(p_rt + p_gb)
    js_div = float(0.5*np.sum(rel_entr(p_rt,m)) + 0.5*np.sum(rel_entr(p_gb,m)))

    ks_stat, _ = ks_2samp(rt, gbsm)   # p-value intentionally discarded
    w1   = wasserstein_distance(rt, gbsm)
    w1_norm = w1 / var_range

    xs  = np.linspace(xlo, xhi, 500)
    cdf_rt  = np.searchsorted(np.sort(rt),   xs, side='right') / len(rt)
    cdf_gb  = np.searchsorted(np.sort(gbsm), xs, side='right') / len(gbsm)
    rmse    = float(np.sqrt(np.mean((cdf_rt - cdf_gb)**2)))
    pear_r, _ = pearsonr(p_rt, p_gb)

    print(f"\n  ┌─ {variable_name} ─────────────────────────────────")
    print(f"  │  KL divergence  : {kl_div:.5f} nats"
          + ("  ✓ excellent" if kl_div<0.05 else ""))
    print(f"  │  JS divergence  : {js_div:.5f} nats")
    print(f"  │  KS statistic   : {ks_stat:.5f}   (p-value omitted: overpowered at n=50k)")
    print(f"  │  Wasserstein-1  : {w1:.6f}")
    print(f"  │  W1 / range     : {w1_norm*100:.3f}%   (dimensionless mismatch)")
    print(f"  │  ECDF RMSE      : {rmse:.5f}")
    print(f"  │  Pearson r(PDF) : {pear_r:.5f}")
    print(f"  └────────────────────────────────────────────────────")

    return dict(Variable=variable_name, KL=round(kl_div,5), JS=round(js_div,5),
                KS_stat=round(ks_stat,5),
                W1=round(w1,6), W1_norm_pct=round(w1_norm*100, 4),
                RMSE_ECDF=round(rmse,5), Pearson_r=round(pear_r,5))


def run_full_validation(rt_arrays, gbsm_arrays):
    print(f"\n{'═'*70}")
    print(f"  QUANTITATIVE VALIDATION  (KL · JS · W1 · ECDF-RMSE · Pearson r)")
    print(f"{'═'*70}")
    results = [
        compute_validation_metrics(rt_arrays['toa'],     gbsm_arrays['toa'],
                                   variable_name="ToA (ns)"),
        compute_validation_metrics(rt_arrays['el'],      gbsm_arrays['el'],
                                   variable_name="AoA Elevation (deg)"),
        compute_validation_metrics(rt_arrays['az'],      gbsm_arrays['az'],
                                   variable_name="AoA Azimuth (deg)"),
        compute_validation_metrics(rt_arrays['doppler'], gbsm_arrays['doppler'],
                                   variable_name="Doppler (Hz)"),
    ]
    print(f"\n  Reporting guidance:")
    print(f"    • W1 / range — primary dimensionless mismatch metric")
    print(f"    • Pearson r > 0.85 — strong distributional shape agreement")
    print(f"    • JS < 0.05 — excellent statistical agreement")
    print(f"    • KS p-value omitted (overpowered at n=50,000)")
    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════

def _apply_kde(samples, weights, eval_pts, bw_value=None):
    """
    Accept an explicit bandwidth so RT and GBSM can be rendered
    at the same kernel width.
    """
    valid = weights > 0
    if not np.any(valid): return np.zeros_like(eval_pts)
    dv, wv = samples[valid], weights[valid]
    ws = wv.sum()
    if ws == 0: return np.zeros_like(eval_pts)
    n_draw = min(8000, len(dv))
    idx    = np.random.choice(len(dv), size=n_draw, p=wv/ws, replace=True)
    sample_arr = dv[idx]
    try:
        if bw_value is None:
            return gaussian_kde(sample_arr, bw_method='scott')(eval_pts)
        std_arr = float(np.std(sample_arr, ddof=1))
        if std_arr <= 0:
            return gaussian_kde(sample_arr, bw_method='scott')(eval_pts)
        return gaussian_kde(sample_arr, bw_method=bw_value/std_arr)(eval_pts)
    except Exception:
        return np.zeros_like(eval_pts)


def plot_rt_vs_gbsm(sim_rt, gbsm, sim_gbsm, scenario_name, filename):
    toa_rt  = np.array([p.toa_ns       for p in sim_rt.paths])
    el_rt   = np.array([p.aoa_elev_deg for p in sim_rt.paths])
    az_rt   = np.array([p.aoa_azim_deg for p in sim_rt.paths])
    dop_rt  = np.array([p.doppler_hz   for p in sim_rt.paths])
    pwr_rt  = _power_linear(sim_rt.paths)
    pwr_rt /= pwr_rt.sum()

    # Get RT's natural Scott's-rule bandwidths, then pass them to
    # GBSM's KDE so both estimators render at identical kernel width.
    xs_toa,  ys_toa,  bw_toa  = smooth_pdf(toa_rt,  weights=pwr_rt)
    xs_elev, ys_elev, bw_elev = smooth_pdf(el_rt,   weights=pwr_rt)
    xs_azim, ys_azim, bw_azim = smooth_pdf(az_rt,   weights=pwr_rt)
    xs_dop,  ys_dop,  bw_dop  = smooth_pdf(dop_rt,  weights=pwr_rt)
    print(f"  RT KDE bandwidths (Scott's rule on RT data — shared with GBSM):")
    print(f"    ToA: {bw_toa:.2f} ns   Elevation: {bw_elev:.2f}°   "
          f"Azimuth: {bw_azim:.2f}°   Doppler: {bw_dop:.2f} Hz")

    W = {k: sim_gbsm[k]['weight'] for k in ('los','sb_rx','sb_tx','db')}
    wt_map = {'los': gbsm.w_los, 'sb_rx': gbsm.w_sb_rx,
              'sb_tx': gbsm.w_sb_tx, 'db': gbsm.w_db}

    tau_ax = np.linspace(180, 270, 800)
    el_ax  = np.linspace(-90, 90, 800)
    az_ax  = np.linspace(-180, 180, 800)
    dop_ax = np.linspace(-1500, 500, 800)

    def gbsm_total(field, ax, bw_value):
        pdf = np.zeros_like(ax, dtype=float)
        for k in ('los','sb_rx','sb_tx','db'):
            d = sim_gbsm[k][field]
            if field == 'toa': d = d*1e9
            pdf += wt_map[k] * _apply_kde(d, W[k], ax, bw_value=bw_value)
        return pdf

    g_toa = gbsm_total('toa',     tau_ax, bw_toa)
    g_el  = gbsm_total('el',      el_ax,  bw_elev)
    g_az  = gbsm_total('az',      az_ax,  bw_azim)
    g_dop = gbsm_total('doppler', dop_ax, bw_dop)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    D    = np.linalg.norm(sim_rt.rx - sim_rt.tx)
    spd  = np.linalg.norm(sim_rt.vel)*3.6
    fig.suptitle(
        f'6G Vehicle Interior Channel — {scenario_name}\n'
        f'Ray Tracer (Ground Truth) vs Hybrid GBSM (Stochastic)\n'
        f'D={D:.0f} m  |  fc=28 GHz  |  v={spd:.0f} km/h',
        fontsize=13, fontweight='bold', y=1.01)

    RT_COL   = '#1a252f'
    GBSM_COL = '#e74c3c'

    def panel(ax, xs_rt, ys_rt, xs_gb, ys_gb, xlabel, title,
              xlim=None, ylim=None):
        if xs_rt is not None:
            ax.fill_between(xs_rt, ys_rt, color='#85929e', alpha=0.18)
            ax.plot(xs_rt, ys_rt, color=RT_COL,   lw=2.8, label='Ray Tracer (Ground Truth)')
        if xs_gb is not None:
            ax.plot(xs_gb, ys_gb, color=GBSM_COL, lw=2.2, ls='--', label='GBSM (Stochastic)')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Power Probability Density', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2, ls='--')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

    panel(axes[0,0], xs_toa, ys_toa, tau_ax, g_toa,
          'Time of Arrival (ns)', 'ToA PDF', xlim=(180,260), ylim=(0,None))
    panel(axes[0,1], xs_elev, ys_elev, el_ax, g_el,
          'Elevation Angle θ (°)', 'AoA Elevation PDF', xlim=(-90,90))
    panel(axes[1,0], xs_azim, ys_azim, az_ax, g_az,
          'Azimuth Angle φ (°)', 'AoA Azimuth PDF', xlim=(-180,180))
    panel(axes[1,1], xs_dop, ys_dop, dop_ax, g_dop,
          'Doppler Shift (Hz)', 'Doppler Spectrum PDF', xlim=(-1500,500))

    legend_lines = [Line2D([0],[0], color=RT_COL,   lw=2.8,       label='Ray Tracer'),
                    Line2D([0],[0], color=GBSM_COL, lw=2.2, ls='--',label='Hybrid GBSM')]
    fig.legend(handles=legend_lines, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5,-0.02))

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✔ Comparison figure saved: {filename}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

# Vehicle-class spec — Tata Starbus Urban City Electric 12m variant.
# Source: published Tata Motors datasheet (CMV360, Tata Motors CV portal).
#   Length : 11.9 m   (datasheet 11900 mm)
#   Width  : 2.58 m   (datasheet 2580 mm)
#   Height : 3.20 m   (typical low-floor city bus exterior height)
#   Glass band lower edge h_glass = 1.0 m (standard window-sill height)
# These are PUBLISHED CLASS specifications — not values extracted from
# the ray-tracer simulation.
BUS_GEOMETRY = dict(
    center  = [50.0, 0.0, 0.0],
    L       = 11.9,      # m  (Tata Starbus Urban 12m)
    W       = 2.58,      # m
    H       = 3.20,      # m
    h_glass = 1.0,       # m  (glass band lower edge)
)

TX_POS  = np.array([0.0,  0.0, 30.0])
RX_POS  = np.array([50.0, 0.0,  1.2])
RX_VEL  = np.array([13.89, 0.0,  0.0])  # 50 km/h


def run_scenario(scenario, num_gbsm_samples=400_000, save_dir="."):
    print(f"\n{'╔'+'═'*68+'╗'}")
    print(f"║  SCENARIO: {scenario.upper():<56}   ║")
    print(f"{'╚'+'═'*68+'╝'}")

    np.random.seed(7)
    bus  = Bus(center=BUS_GEOMETRY['center'],
               length=BUS_GEOMETRY['L'],
               width=BUS_GEOMETRY['W'],
               height=BUS_GEOMETRY['H'])
    bldgs = make_buildings(scenario, seed=42)
    sim_rt = RayTracer6G(TX_POS, RX_POS, RX_VEL, bldgs, bus)
    sim_rt.run(f"{scenario.title()} Urban | 28 GHz")

    if len(sim_rt.paths) == 0:
        print("  [!] No RT paths found — skipping scenario.")
        return

    np.random.seed(42)
    params = extract_params_from_rt_paths(sim_rt.paths, TX_POS, RX_POS)

    gbsm = HybridGBSM_Rigorous(
        tx_pos          = TX_POS,
        rx_pos          = RX_POS,
        bus_geometry    = BUS_GEOMETRY,
        R1              = params['R1'],
        R2              = params['R2'],
        lambda_excl     = params['lambda_excl'],
        kappa           = params['kappa'],
        mu_hat          = params['mu_hat'],
        w_los           = params['w_los'],
        w_sb_rx         = params['w_sb_rx'],
        w_sb_tx         = params['w_sb_tx'],
        w_db            = params['w_db'],
        use_mixture_vmf        = True,
        street_azimuth_peaks   = params['az_peaks'],
        interior_phi_peaks_deg = params['interior_phi_peaks_deg'],
        kappa_int              = params['kappa_int'],
        interior_dB_per_m      = INTERIOR_LOSS_PER_M,
        glass_dB               = GLASS.transmission_loss_db,
    )

    sim_gbsm = gbsm.simulate_all_paths(num_samples=num_gbsm_samples)

    rt_arrays   = extract_rt_sample_arrays(sim_rt.paths, n_samples=50_000)
    model_wts   = {'w_los': gbsm.w_los, 'w_sb_rx': gbsm.w_sb_rx,
                   'w_sb_tx': gbsm.w_sb_tx, 'w_db': gbsm.w_db}
    gbsm_arrays = aggregate_gbsm_samples(sim_gbsm, model_wts, n_samples=50_000)

    val_results = run_full_validation(rt_arrays, gbsm_arrays)

    fig_name = os.path.join(save_dir, f"rt_vs_gbsm_{scenario}.png")
    plot_rt_vs_gbsm(sim_rt, gbsm, sim_gbsm, scenario.title(), fig_name)

    return {'rt': sim_rt, 'gbsm': gbsm, 'sim': sim_gbsm,
            'rt_arrays': rt_arrays, 'gbsm_arrays': gbsm_arrays,
            'validation': val_results, 'params': params}


def main():
    out_dir = os.getcwd()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  INTEGRATED 6G CHANNEL MODEL v11 — Ray Tracer + Hybrid GBSM     ║")
    print("║  APMC 2026 | Rishab Rao (21D070056)                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    dense_out  = run_scenario('dense',  num_gbsm_samples=400_000, save_dir=out_dir)
    sparse_out = run_scenario('sparse', num_gbsm_samples=400_000, save_dir=out_dir)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  ALL SCENARIOS COMPLETE                                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

# Install and load the package
install.packages("rvMF")
library(rvMF)

# Generate 100 samples in 3D (p=3) with concentration kappa=10
samples <- rvMF(n = 100, mu = c(0, 0, 1), kappa = 10)
