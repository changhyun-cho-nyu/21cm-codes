"""
prepare_orcd.py
===============
Prepares one floor-box variant (see make_variants.py for the idea) to be built from the AREPO source the 50 cMpc
X-ray runs were built from (SonjaXrayPaper/withXrays/arepo; its Config.sh is flag-for-flag Lumina's snapshot /Config),
on ORCD. Writes into <out>:
  Config.sh       Lumina's own build Config minus radiative transfer, galaxy physics and output-only flags (list in
                  REMOVE), plus GENERATE_GAS_IN_ICS, PMGRID scaled to the box, and the variant's changes;
  param.txt       exactly the tags this Config requires in the old parameters.cc (it stops on unknown or missing tags),
                  every value from the run's parameters-usedvalues except the overrides in OVERRIDES / per variant;
  output_list.txt z = 45 40 36 33 30 27 24 21 18 15 12;
  ics.hdf5        dark-matter-only Zel'dovich initial conditions at z = 49 (the old source has no N-GenIC); the same
                  white noise for every variant, and the N = 128 grid shares the N = 64 grid's Fourier modes.
                  P(k): Eisenstein & Hu (1998) no-wiggle transfer function, n_s = 0.9665, sigma_8 = 0.8102 (Planck 2018,
                  Lumina's cosmology); growth from the linear growth equation with radiation (D(1)/D(0.02) = 38.864; N-GenIC 38.868); velocities sqrt(a) H(a) f(a) D psi (AREPO/GADGET IC
                  convention, H with radiation as INCLUDE_RELATIVISTIC_OMEGAS).

  python prepare_orcd.py variant --variant fid --arepo <copy of the source> --config <run Config.sh>
                                 --params <run parameters-usedvalues> --out <dir>
  python prepare_orcd.py check-tags --arepo <src> --config <run Config.sh> --params <run parameters-usedvalues>
      (validation: the tags required by the run's own Config must equal the run's used-parameter list)
"""
import argparse
import os
import re
import sys

import numpy as np

SPACING = 56.38                       # ckpc/h: Lumina's mean gas-cell spacing (33830 / 600)
ZOUT = (45, 40, 36, 33, 30, 27, 24, 21, 18, 15, 12)
NMAX = 128                            # white-noise grid shared by all variants
SEED = 20261125
# removed from Lumina's Config: radiative transfer, galaxy physics, group finding, outputs, restarts and memory
# layouts for the 6000^3 run (none acts on cold neutral gas before sources form). SHAREDNODEMEMORY is kept (2026-11-27):
# the kept HIERARCHICAL_DOMAIN_DECOMPOSITION needs it (domain.cc: "Not implemented" without it, and SMM is declared only
# with it; job 30722651 failed there); both are parallelisation flags, as in the run's own Config
REMOVE_PREFIX = ("MRT", "GFM", "BH_", "FOF", "SUBFIND", "REIONIZATION_", "POWERSPEC_ON_OUTPUT")
REMOVE = {"COOLING", "UVB_OFF", "USE_SFR", "SOFTEREQS", "STEEPER_SFR_FOR_STARBURST", "BLACK_HOLES", "DRAINGAS",
          "MERGERTREE", "CUDA", "AMD_HIP", "EOR_LOG", "RT_LOG", "RENDERGRID", "CONEGRID", "PROJECTION_GRID",
          "MTNG", "RESTART_XL", "MULTIPLE_RESTARTS", "HIERARCHICAL_RESTARTS", "MEMORY_SAVING_XL",
          "VORONOI_SEARCH_XL", "EVALPOTENTIAL", "OUTPUT_COORDINATES_AS_INTEGERS", "OUTPUT_VELOCITIES_IN_HALF_PRECISION",
          "DEBUG_ENABLE_FPU_EXCEPTIONS"}
ADD = {"GENERATE_GAS_IN_ICS": ""}
FLOOR = ("ENFORCE_JEANS_STABILITY_OF_CELLS", "ENFORCE_JEANS_STABILITY_OF_CELLS_EEOS")
ENTROPY = {"USE_ENTROPY_FOR_COLD_FLOWS": "", "ENTROPY_MACH_THRESHOLD": "1.1"}
# The entropy switch needs the exact Riemann solver in this source: it keeps the entropy where MaxMach < 1.1
# (update_primitive_variables.cc; without ENTROPY_MACH_THRESHOLD it stops, "not implemented any more"), MaxMach comes
# from get_mach_numbers, and that is defined only in riemann.cc, which compiles only without RIEMANN_HLLC/HLL/ROSUNOV/
# HLLD/GAMMA (jobs 30726810_3/4/6: undefined get_mach_numbers). So the entropy variants drop the run's RIEMANN_HLLC
# (exact solver), and fid_exact (fid with the exact solver) separates the solver's effect from the switch's
# (user, 2026-11-27: option A).
EXACT_SOLVER_REMOVES = ("RIEMANN_HLLC",)
VARIANTS = {   # name: (floor, entropy switch, MinGasTemp [K], grid factor, exact Riemann solver)
    "fid": (True, False, 5.0, 1, False), "nofloor": (False, False, 5.0, 1, False),
    "noclamp": (True, False, 0.01, 1, False), "entropy": (True, True, 5.0, 1, True),
    "clean": (False, True, 0.01, 1, True), "fid_hr": (True, False, 5.0, 2, False),
    "clean_hr": (False, True, 0.01, 2, True), "fid_exact": (True, False, 5.0, 1, True)}


# ---------------------------------------------------------------- Config
def read_config(path):
    cfg = {}
    for line in open(path):
        s = line.split("#")[0].strip()
        if not s:
            continue
        k, _, v = s.split()[0].partition("=")
        cfg[k] = v
    return cfg


def variant_config(run_cfg, variant, n):
    floor, entropy, _, _, exact = VARIANTS[variant]
    if entropy and not exact:
        sys.exit(f"STOP: variant {variant}: the entropy switch needs the exact Riemann solver in this source")
    cfg = {k: v for k, v in run_cfg.items() if k not in REMOVE and not k.startswith(REMOVE_PREFIX)}
    if exact:
        for k in EXACT_SOLVER_REMOVES:
            cfg.pop(k, None)
    for k in FLOOR:
        cfg.pop(k, None)
        if floor:
            cfg[k] = ""
    if entropy:
        cfg.update(ENTROPY)
    cfg.update(ADD)
    cfg["PMGRID"] = str(2 * n)
    return cfg


def write_config(cfg, path):
    with open(path, "w") as f:
        for k in sorted(cfg):
            f.write(f"{k}={cfg[k]}\n" if cfg[k] != "" else f"{k}\n")


# ---------------------------------------------------------------- parameter tags of the old parameters.cc
DERIVED = {"REFINEMENT": ("REFINEMENT_SPLIT_CELLS", "REFINEMENT_MERGE_CELLS")}   # allvars.h: defined if any of these


def defines(cfg):
    d = dict(cfg)
    for k, srcs in DERIVED.items():
        if any(s in d for s in srcs):
            d[k] = ""
    return d


def eval_cond(expr, d):
    """Evaluate a preprocessor condition: defined(X), !, &&, ||, comparisons with Config values."""
    e = expr.split("//")[0].split("/*")[0]
    e = re.sub(r"defined\s*\(\s*([A-Za-z_]\w*)\s*\)", lambda m: " True " if m.group(1) in d else " False ", e)
    e = re.sub(r"defined\s+([A-Za-z_]\w*)", lambda m: " True " if m.group(1) in d else " False ", e)
    e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ").replace(" not =", "!=")

    def ident(m):
        w = m.group(0)
        if w in ("True", "False", "and", "or", "not"):
            return w
        v = d.get(w)
        if v is None:
            return "0"
        return v if v not in ("",) else "1"
    e = re.sub(r"[A-Za-z_]\w*", ident, e)
    try:
        return bool(eval(e, {}, {}))
    except Exception:
        sys.exit(f"STOP: cannot evaluate preprocessor condition '{expr.strip()}' -> '{e.strip()}'")


def required_tags(pcc, d):
    """Tags active in read_parameter_file for these defines, in order."""
    text = re.sub(r"\\\s*\n", " ", open(pcc).read())             # join continued preprocessor lines
    i0 = text.find("void read_parameter_file")
    lines = text[i0:].splitlines()
    stack = []        # (this branch active, any branch taken, parent active)
    tags = []
    loop_bound = "0"
    for line in lines:
        s = line.strip()
        act = all(x[0] for x in stack)
        if s.startswith("#ifdef"):
            stack.append([act and s.split()[1] in d, s.split()[1] in d, act])
        elif s.startswith("#ifndef"):
            stack.append([act and s.split()[1] not in d, s.split()[1] not in d, act])
        elif s.startswith("#if"):
            c = eval_cond(s[3:], d)
            stack.append([act and c, c, act])
        elif s.startswith("#elif"):
            top = stack[-1]
            c = (not top[1]) and eval_cond(s[5:], d)
            top[0], top[1] = top[2] and c, top[1] or c
        elif s.startswith("#else"):
            top = stack[-1]
            top[0] = top[2] and not top[1]
            top[1] = True
        elif s.startswith("#endif"):
            stack.pop()
        else:
            fl = re.match(r"for\s*\(\s*i\s*=\s*0\s*;\s*i\s*<\s*([A-Za-z_]\w*|\d+)\s*;", s)
            if fl:
                loop_bound = fl.group(1)
            m = re.search(r'strcpy\(\s*tag\[nt\]\s*,\s*"([A-Za-z0-9_]+)"\s*\)', s)
            if m and act and m.group(1) not in tags:
                tags.append(m.group(1))
            m = re.search(r'sprintf\(\s*buf\s*,\s*"([A-Za-z0-9_]+)%d"\s*,\s*i\s*\)', s)   # tags made in a loop
            if m and act:
                nb = int(loop_bound) if loop_bound.isdigit() else int(d.get(loop_bound) or 0)
                if nb == 0:
                    sys.exit(f"STOP: loop bound {loop_bound} for tag {m.group(1)}%d not in the Config")
                tags += [f"{m.group(1)}{i}" for i in range(nb) if f"{m.group(1)}{i}" not in tags]
        if s.startswith("for(i = 0; i < nt; i++)") or "Parameter file %s not found" in s:
            pass
        if re.search(r'fopen\(fname, "r"\)', s):
            break                     # the tag table ends where the file is read
    return tags


def read_used(path):
    vals = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 2 and not line.startswith("%"):
            vals[p[0]] = p[1]
    return vals


def write_params(tags, used, over, path):
    missing = [t for t in tags if t not in over and t not in used]
    if missing:
        sys.exit(f"STOP: no value for required tags {missing}")
    with open(path, "w") as f:
        for t in tags:
            f.write(f"{t:40s} {over.get(t, used.get(t))}\n")


# ---------------------------------------------------------------- initial conditions
def cosmology(used):
    om, ol, ob, h = (float(used[k]) for k in ("Omega0", "OmegaLambda", "OmegaBaryon", "HubbleParam"))
    tcmb = float(used.get("CMBTemperature", 2.7255))
    orad = 1.0 - om - ol                       # what INCLUDE_RELATIVISTIC_OMEGAS adds (OmegaTot = 1 to 7e-7)
    return om, ol, ob, h, tcmb, orad


def growth(a, om, ol, orad, nstep=200000):
    """Linear growth of matter with radiation: D'' + (2 + dlnE/dlna) D' = 1.5 Omega_m(a) D in ln a (RK4), from the
    Meszaros growing mode D = 1 + 1.5 a/a_eq at a = 1e-6. Returns D(a)/D(1) and f(a) = dlnD/dlna.
    (Check: D(1)/D(0.02) = 38.864 for Lumina's cosmology; AREPO's N-GenIC prints Dplus = 38.868.)"""
    aeq = orad / om
    E2 = lambda x: om / x**3 + orad / x**4 + ol                                         # noqa: E731

    def rhs(x, D, dD):
        aa = np.exp(x)
        dlnE = 0.5 * (-3 * om / aa**3 - 4 * orad / aa**4) / E2(aa)
        return dD, -(2 + dlnE) * dD + 1.5 * om / aa**3 / E2(aa) * D

    def run(x1):
        x0 = np.log(1e-6)
        D, dD = 1 + 1.5e-6 / aeq, 1.5e-6 / aeq
        h = (x1 - x0) / nstep
        x = x0
        for _ in range(nstep):
            k1 = rhs(x, D, dD)
            k2 = rhs(x + h / 2, D + h / 2 * k1[0], dD + h / 2 * k1[1])
            k3 = rhs(x + h / 2, D + h / 2 * k2[0], dD + h / 2 * k2[1])
            k4 = rhs(x + h, D + h * k3[0], dD + h * k3[1])
            D, dD = D + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), dD + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            x += h
        return D, dD / D
    Da, fa = run(np.log(a))
    D1, _ = run(0.0)
    return Da / D1, fa


def eh_nowiggle(k, om, ob, h, tcmb):
    """Eisenstein & Hu (1998) eqs. 26-31, k in h/Mpc."""
    th = tcmb / 2.7
    wm, fb = om * h * h, ob / om
    s = 44.5 * np.log(9.83 / wm) / np.sqrt(1.0 + 10.0 * (ob * h * h)**0.75)
    ag = 1.0 - 0.328 * np.log(431.0 * wm) * fb + 0.38 * np.log(22.3 * wm) * fb**2
    geff = om * h * (ag + (1.0 - ag) / (1.0 + (0.43 * k * h * s)**4))
    q = k * th**2 / geff
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def pk_z0(om, ob, h, tcmb, ns=0.9665, s8=0.8102):
    """P(k) [(Mpc/h)^3] at z = 0, normalised to sigma_8."""
    kk = np.logspace(-5, 3, 20000)
    raw = kk**ns * eh_nowiggle(kk, om, ob, h, tcmb)**2
    x = kk * 8.0
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    s2 = np.trapezoid(raw * W**2 * kk**3 / (2 * np.pi**2), np.log(kk))
    A = s8**2 / s2
    return lambda k: A * np.maximum(k, 1e-30)**ns * eh_nowiggle(np.maximum(k, 1e-30), om, ob, h, tcmb)**2


def make_ics(n, box, used, path, seed=SEED, nmax=NMAX):
    import h5py
    om, ol, ob, h, tcmb, orad = cosmology(used)
    a = float(used["TimeBegin"])
    Dr, f = growth(a, om, ol, orad)                              # D(a)/D(1)
    P = pk_z0(om, ob, h, tcmb)
    Lmpc = box / 1000.0                                         # Mpc/h
    rng = np.random.default_rng(seed)
    W = np.fft.rfftn(rng.standard_normal((nmax, nmax, nmax)))   # <|W|^2> = nmax^3
    ki = np.fft.fftfreq(n, 1.0 / n).astype(int)                 # integer wavenumbers of the n grid
    kz = np.arange(n // 2 + 1)
    sub = W[np.ix_(ki % nmax, ki % nmax, kz)]                   # the same modes on both grids
    kx, ky, kzz = np.meshgrid(ki, ki, kz, indexing="ij")
    kf = 2 * np.pi / Lmpc
    k = kf * np.sqrt(kx**2 + ky**2 + kzz**2)
    amp = np.sqrt(n**6 * P(k) / (Lmpc**3 * nmax**3)) * Dr
    nyq = (np.abs(kx) == n // 2) | (np.abs(ky) == n // 2) | (kzz == n // 2) | (k == 0)
    delta = np.where(nyq, 0.0, sub * amp)
    psi = []
    for kc in (kx, ky, kzz):
        comp = np.where(k > 0, 1j * kc * kf * delta / np.where(k > 0, k, 1.0)**2, 0.0)
        psi.append(np.fft.irfftn(comp, s=(n, n, n), axes=(0, 1, 2)) * 1000.0)   # displacement D psi in kpc/h
    psi = np.stack([p.ravel() for p in psi], axis=1)
    q = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) * (box / n)
    pos = np.mod(q + psi, box)
    Hcode = 0.1 * np.sqrt(om / a**3 + orad / a**4 + ol)         # km/s per kpc/h
    vel = np.sqrt(a) * Hcode * f * psi
    rhocrit = 3 * 0.1**2 / (8 * np.pi * 43018.7)                # code units (1e10 Msun/h, kpc/h)
    mdm = om * rhocrit * box**3 / n**3                          # total matter: GENERATE_GAS_IN_ICS splits off the gas
    with h5py.File(path, "w") as fo:
        hd = fo.create_group("Header")
        npt = np.array([0, n**3, 0, 0, 0, 0], dtype=np.uint32)
        hd.attrs.update(dict(NumPart_ThisFile=npt, NumPart_Total=npt, NumPart_Total_HighWord=np.zeros(6, np.uint32),
                             MassTable=np.array([0, mdm, 0, 0, 0, 0], dtype=np.float64), Time=a, Redshift=1 / a - 1,
                             BoxSize=float(box), NumFilesPerSnapshot=1, Omega0=om, OmegaLambda=ol, HubbleParam=h,
                             Flag_Sfr=0, Flag_Cooling=0, Flag_StellarAge=0, Flag_Metals=0, Flag_Feedback=0,
                             Flag_DoublePrecision=1))
        g = fo.create_group("PartType1")
        g["Coordinates"] = pos
        g["Velocities"] = vel
        g["ParticleIDs"] = np.arange(1, n**3 + 1, dtype=np.uint64)
    return dict(a=a, D_ratio=1.0 / Dr, f=f, vel_prefac=np.sqrt(a) * Hcode * f, mdm=mdm,
                psi_rms=float(np.sqrt((psi**2).sum(1).mean())), psi_max=float(np.abs(psi).max()))


def output_list(path, zend=12.0):
    with open(path, "w") as f:
        for z in ZOUT:
            if z >= zend - 1e-9:
                f.write(f"{1.0 / (1.0 + z):.8f} 1\n")


# ---------------------------------------------------------------- commands
def cmd_variant(a):
    floor, entropy, tmin, fac, exact = VARIANTS[a.variant]
    n = a.n * fac
    box = SPACING * a.n
    os.makedirs(a.out, exist_ok=True)
    cfg = variant_config(read_config(a.config), a.variant, n)
    write_config(cfg, os.path.join(a.out, "Config.sh"))
    used = read_used(a.params)
    tags = required_tags(os.path.join(a.arepo, "src", "parameters.cc"), defines(cfg))
    over = dict(InitCondFile="ics", OutputDir="output", OutputListFilename="output_list.txt", BoxSize=f"{box:.4f}",
                TimeMax=f"{1.0 / (1.0 + a.zend):.8f}", MinGasTemp=str(tmin), NumFilesPerSnapshot="1",
                MaxFilesWithConcurrentIO="1", TimeLimitCPU=str(a.cpu_limit), CpuTimeBetRestartFile="3600",
                MaxMemSize=str(a.maxmem), MaxMemSizeNode=str(a.maxmem_node), CoolingOn="0", StarformationOn="0", ResubmitCommand="none",
                OutputListOn="1", TimeBetSnapshot="0", TimeOfFirstSnapshot="0")
    write_params(tags, used, over, os.path.join(a.out, "param.txt"))
    output_list(os.path.join(a.out, "output_list.txt"), a.zend)
    info = make_ics(n, box, used, os.path.join(a.out, "ics.hdf5"))
    print(f"{a.variant}: N = {n}^3, box {box:.2f} ckpc/h, floor {floor}, entropy switch {entropy}, MinGasTemp {tmin} K, "
          f"Riemann solver {'exact' if exact else 'HLLC'}")
    print(f"  Config: {len(cfg)} flags (run Config {len(read_config(a.config))}); param.txt: {len(tags)} tags")
    print(f"  ICs: a = {info['a']}, D(1)/D(a) = {info['D_ratio']:.3f}, f = {info['f']:.4f}, velocity prefactor "
          f"{info['vel_prefac']:.4f}, particle mass {info['mdm']:.6g} (x {1 - float(used['OmegaBaryon']) / float(used['Omega0']):.4f} "
          f"DM after the gas split), displacement rms {info['psi_rms']:.2f} kpc/h ({info['psi_rms'] / (box / n):.3f} spacings), "
          f"max {info['psi_max']:.2f}")


def cmd_check_tags(a):
    run_cfg = read_config(a.config)
    tags = required_tags(os.path.join(a.arepo, "src", "parameters.cc"), defines(run_cfg))
    used = list(read_used(a.params))
    print(f"run Config: {len(tags)} required tags; parameters-usedvalues: {len(used)} tags")
    print("  required but not in usedvalues:", [t for t in tags if t not in used])
    print("  in usedvalues but not required:", [t for t in used if t not in tags])
    for v in VARIANTS:
        cfg = variant_config(run_cfg, v, 64)
        t2 = required_tags(os.path.join(a.arepo, "src", "parameters.cc"), defines(cfg))
        new = [t for t in t2 if t not in used]
        print(f"  {v}: {len(t2)} tags; not in usedvalues: {new}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("variant")
    v.add_argument("--variant", choices=list(VARIANTS), required=True)
    v.add_argument("--n", type=int, default=64)
    v.add_argument("--zend", type=float, default=12.0)
    v.add_argument("--maxmem", type=int, default=4000)
    v.add_argument("--maxmem-node", type=int, default=4000,
                   help="MaxMemSizeNode [MB], the SHAREDNODEMEMORY pool allocated once per node (run: 7000); "
                        "with 32 tasks x --maxmem 2500 it keeps the job inside --mem=100G")
    v.add_argument("--cpu-limit", type=int, default=82800)
    for s in (v, sp.add_parser("check-tags")):
        s.add_argument("--arepo", required=True)
        s.add_argument("--config", required=True)
        s.add_argument("--params", required=True)
    v.add_argument("--out", required=True)
    a = p.parse_args()
    {"variant": cmd_variant, "check-tags": cmd_check_tags}[a.cmd](a)


if __name__ == "__main__":
    main()
