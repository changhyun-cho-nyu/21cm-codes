"""
test_cells_Tb.py
================
delta T_b from the Voronoi gas cells of a full Lumina snapshot, against the production calculation from the
640^3 render: the resolution (clumping) test of the 3.46 cm signal, and of 21 cm at high z, at the simulation's
own resolution.

Why. The maps compute delta T_b from pixel averages (mass-weighted T and fractions, mean density). delta T_b^He
~ n_e n_HeII f(T) is two-body, so the average of the cell values exceeds the value of the averages; the render
test (test_resolution_Tb.py) found no convergence up to 2560^3. Here the limit is the gas cells themselves.

Cells (render conventions, RT_movie_3D_rendering.cc): star-forming gas neutral with He II = He III = 0; T from u
and Ne (Ne = HII + HeII + 2 HeIII, as AREPO-RT sets it); J_alpha of the 640^3 file at the cell's pixel; delta T_b
with tb_physics.py (the physics of the C codes). The repackaged snapshots are not in spatial order (inspect), so:

  scan   (array job) each task reads all fields of its share of the rows and
         - adds the cells into 640^3 sums (volume, volume x dT_b, mass, mass x T, x_HII, f_HeII), whole box;
         - keeps the cells of --n-cube random cubes (--cube-side^3 of the 320^3 pixels, aligned, no overlap).
         The render axis order is found from the data (corr of cell density with the render Density).
  merge  sums the tasks and reports
         - checks: the cells deposited at 640^3 against the render Density, T, x_HII, f_HeII;
         - WHOLE BOX: <dT_b> of the production calculation (render inputs) vs the cell level (exact: volume-
           weighted mean over all cells), and P(k), r(k) of the production map vs the cell-level map at 640^3
           (each pixel the volume-weighted mean of its cells' dT_b);
         - CUBES: the ladder, cells grouped by centre into 320^3 ... 10240^3 pixels (density sum(m)/sum(V),
           mass-weighted T and fractions, volume-weighted J), dT_b of the averages; ratio to the cell level with a
           jackknife over cubes; variance of the maps smoothed to 320^3 and 640^3 pixels.

Usage
-----
  python test_cells_Tb.py inspect --snapdir <.../snapshots> [--snap 92] [--renders A B] [--jdir <Jalpha>]
  python test_cells_Tb.py scan  --snapdir D --snap 136 --line He --J <J stars> --J <J AGN> --render <ren_640> \
         --render-snap 501 --out <dir> --task 0 --ntask 16
  python test_cells_Tb.py merge --snapdir D --snap 136 --line He --J ... --J ... --render ... --render-snap 501 --out <dir>
Full snapshots that fall between renders (11, 17, 24: |dz| = 0.03-0.07): --render-snap A (nearest; the pixel
reference) with --max-dz above |dz|, plus --J-b <files> --render-snap-b B (the other side): J linear in z between A
and B at the snapshot's z, for the cells and the reference alike (user decision 2026-11-17). Mock-tested with a
planted answer (5e-7, float32) and the refusals (no bracket, J file at another z, merge/scan J mismatch).
Memory: scan ~20 GB (six 640^3 float64 sums, J, chunk); merge ~40 GB. Disk: ~6 GB per task plus the cube cells
(~45 B per cell; 64 cubes of 16^3 pixels ~ 0.8% of the cells, ~75 GB).
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from tb_physics import SRC, X_H, Y_HE, M_H, G, MPC, OMEGA_L_DEFAULT, Line
from check_tb_choices import spectra

# AREPO constants (RT_movie_3D_rendering.cc uses them for the render temperature)
GAMMA_MINUS1, BOLTZMANN, PROTONMASS, HYDROGEN_MASSFRAC = 2.0 / 3.0, 1.38065e-16, 1.67262178e-24, 0.76
GAS_FIELDS = ("Coordinates", "Masses", "Density", "InternalEnergy", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction",
              "StarFormationRate")
# Lumina's repackaged snapshots store IntCoordinates (32-bit integers, x = int * BoxSize / 2^32, Zier+26 Sec. 9.1)
# and no ElectronAbundance. AREPO-RT always sets Ne = HII + HeII + 2 HeIII (MRT/RT_update_primitive_variables.cc,
# MRT/RT.cc; star-forming gas: set_full_ionization_mrt, Ne = (1+X)/2X = 1 + 2y), so Ne is rebuilt from the stored
# fractions (before the render's neutral convention for star-forming gas), as the render used it for T.
LEVEL_FACTORS = (0.5, 1, 2, 4, 8, 16)   # ladder grids in units of the render grid N: 320^3 ... 10240^3 for N = 640


def levels(N):
    return [int(f * N) for f in LEVEL_FACTORS]


class Snapshot:
    """header and PartType0 fields of a full snapshot (columnar layout or the snap_NNN.hdf5 virtual file)"""
    def __init__(self, snapdir, snap, gas_file=None):
        self.path = os.path.join(snapdir, f"snap_{snap:03d}.hdf5")
        if not os.path.exists(self.path):
            sys.exit(f"ERROR: no {self.path}")
        with h5py.File(self.path, "r") as f:
            self.hd = {}
            for grp in ("Parameters", "Config", "Header"):          # Header last: it wins
                if grp in f:
                    self.hd.update({k: v for k, v in f[grp].attrs.items()})
        cands = [gas_file] if gas_file else [os.path.join(snapdir, "PartType0", f"PartType0_{snap:03d}.hdf5"), self.path]
        self.f, self.g = None, None
        for c in cands:
            if c and os.path.exists(c):
                f = h5py.File(c, "r")
                g = f["PartType0"] if "PartType0" in f else f
                if "Density" in g:
                    self.f, self.g, self.gas_path = f, g, c
                    break
                f.close()
        if self.g is None:
            sys.exit(f"ERROR: no PartType0 fields in {cands}")
        self.pos_name = "Coordinates" if "Coordinates" in self.g else "IntCoordinates"
        self.has_ne = "ElectronAbundance" in self.g
        missing = [k for k in GAS_FIELDS if k != "Coordinates" and k not in self.g]
        if self.pos_name not in self.g:
            missing.append("Coordinates / IntCoordinates")
        if missing:
            print(f"ERROR: {self.gas_path} lacks {missing}. What is there:")
            print(f"  {self.gas_path} [{self.g.name}]: " + ", ".join(
                f"{k} {self.g[k].shape if isinstance(self.g[k], h5py.Dataset) else '(group)'}" for k in self.g))
            d0 = os.path.join(snapdir, "PartType0")
            if os.path.isdir(d0):
                ents = sorted(os.listdir(d0))
                mine = [e for e in ents if f"{snap:03d}" in e] or ents[:20]
                print(f"  {d0}: {len(ents)} entries; for snapshot {snap}: {mine[:40]}")
                for e in mine[:3]:
                    q = os.path.join(d0, e)
                    if os.path.isfile(q) and e.endswith(".hdf5"):
                        with h5py.File(q, "r") as f:
                            f.visititems(lambda n, o: print(f"    {e}:{n} {o.shape} {o.dtype}")
                                         if isinstance(o, h5py.Dataset) else None)
                    elif os.path.isdir(q):
                        print(f"    {e}/: {sorted(os.listdir(q))[:10]}")
            sys.exit(1)
        self.n = self.g["Density"].shape[0]
        self.z = float(self.hd["Redshift"]) if "Redshift" in self.hd else 1.0 / float(self.hd["Time"]) - 1.0
        self.box = float(self.hd["BoxSize"])
        self.h = float(self.hd["HubbleParam"])
        pd = self.g[self.pos_name]
        if np.issubdtype(pd.dtype, np.integer) and not np.issubdtype(pd.dtype, np.unsignedinteger):
            sys.exit(f"STOP: {self.pos_name} is {pd.dtype} (signed); the int -> position mapping for signed integers is "
                     f"not documented. Report the dtype and min/max of a few rows before going on.")

    def unit(self, name, default):
        return float(self.hd.get(name, default))

    def read(self, name, i0, i1):
        if name == "Coordinates":
            d = self.g[self.pos_name]
            a = d[i0:i1].astype(np.float64)
            if np.issubdtype(d.dtype, np.integer):
                a *= self.box / 2.0**int(8 * d.dtype.itemsize)        # unsigned 32-bit: int * BoxSize / 2^32
            return a
        if name == "ElectronAbundance" and not self.has_ne:
            return (self.g["HII_Fraction"][i0:i1].astype(np.float64) + self.g["HeII_Fraction"][i0:i1].astype(np.float64)
                    + 2.0 * self.g["HeIII_Fraction"][i0:i1].astype(np.float64))
        return self.g[name][i0:i1].astype(np.float64)


def first_3d(f):
    return f[next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)]


def load_J(files, N):
    """sum of the 640^3 J_alpha files (stars, AGN); --J none alone means J = 0 (no J files: He above z = 10.01,
    where the Lyman-alpha share of the coupling is <= 0.4%, JALPHA_NOTES 4.7)"""
    if list(files) == ["none"]:
        print("J = 0 (--J none)")
        return np.zeros((N,) * 3, np.float32)
    return sum(first_3d(h5py.File(fn, "r"))[()].astype(np.float32) for fn in files)


def render_z(render, snap):
    with h5py.File(os.path.join(render, "All", f"All_{snap:03d}.hdf5"), "r") as f:
        return float(f["Header"].attrs["Redshift"])


def check_J_z(files, zr, tag):
    """the J files' own Redshift attribute (written by the J codes) must be the render's z"""
    if list(files) == ["none"]:
        return
    for fn in files:
        with h5py.File(fn, "r") as f:
            zj = first_3d(f).attrs.get("Redshift")
        if zj is None:
            print(f"  note: {fn} has no Redshift attribute (not checked against render {tag})")
        elif abs(float(zj) - zr) > 1e-4:
            sys.exit(f"STOP: {fn} is at z = {float(zj)}, render {tag} at z = {zr}")


TAG_NEW = ("spin_model", "J_files", "J_b_files", "ntask")      # task-file attributes written from 2026-11-19 on


def _same(got, want):
    if isinstance(got, bytes):
        got = got.decode()
    if isinstance(want, str):
        return str(got) == want
    if isinstance(want, (list, tuple)):
        return [float(x) for x in np.atleast_1d(got)] == [float(x) for x in want]
    return float(got) == float(want)


def check_tag(fa, a, fn, extra=None):
    """a task file must come from the same line, spin model, render(s) and J files as this run (scan reuse, merge);
    audit 2026-11-19: before, a rescan reused any existing task file and the merge checked only the renders.
    Task files written before 2026-11-19 have no TAG_NEW attributes: those are not checked (a note is printed)."""
    want = {"line": a.line, "spin_model": a.spin_model, "render_snap": a.render_snap, "J_interp": 1 if a.J_b else 0,
            "J_files": ";".join(os.path.basename(x) for x in a.J), "J_b_files": ";".join(os.path.basename(x) for x in a.J_b)}
    if a.J_b:
        want["render_snap_b"] = a.render_snap_b
    want.update(extra or {})
    bad, unchecked = [], []
    for k, v in want.items():
        got = fa.get(k, 0 if k == "J_interp" else None)
        if got is None:
            (unchecked if k in TAG_NEW else bad).append(k)
        elif not _same(got, v):
            bad.append(f"{k} = {got!r} (this run: {v!r})")
    if bad:
        sys.exit(f"ERROR: {fn} was made with different settings: {'; '.join(map(str, bad))}. Remove it (or use another --out).")
    return unchecked


def J_at(a, N, z):
    """J at the snapshot's z: the --J files (render --render-snap), or, with --J-b, linear in z between render
    --render-snap (a) and --render-snap-b (b), which must bracket z (user decision 2026-11-17 for the full
    snapshots 11, 17, 24, which fall between renders)"""
    za = render_z(a.render, a.render_snap)
    if abs(za - z) > a.max_dz:
        sys.exit(f"ERROR: render snapshot {a.render_snap} z = {za} vs snapshot z = {z} (|dz| > --max-dz {a.max_dz})")
    if not a.J_b:
        check_J_z(a.J, za, f"{a.render_snap}")                    # audit 2026-11-19: was checked only with --J-b
        J = load_J(a.J, N)
        return J, {"J_interp": 0}
    if a.render_snap_b is None or list(a.J) == ["none"] or list(a.J_b) == ["none"]:
        sys.exit("ERROR: --J-b needs --render-snap-b and J files on both sides")
    zb = render_z(a.render, a.render_snap_b)
    if not (min(za, zb) < z < max(za, zb)):
        sys.exit(f"STOP: snapshot z = {z} is not strictly between render {a.render_snap} (z = {za}) and "
                 f"{a.render_snap_b} (z = {zb})")
    check_J_z(a.J, za, f"{a.render_snap}")
    check_J_z(a.J_b, zb, f"{a.render_snap_b}")
    w = (z - za) / (zb - za)                                      # weight of render b, linear in z
    Ja = load_J(a.J, N)
    Jb = load_J(a.J_b, N)
    ma, mb = float(Ja.mean(dtype=np.float64)), float(Jb.mean(dtype=np.float64))
    J = ((1.0 - w) * Ja + w * Jb).astype(np.float32)
    del Ja, Jb
    print(f"J linear in z: render {a.render_snap} (z = {za:.5f}, <J> {ma:.4e}) and {a.render_snap_b} (z = {zb:.5f}, "
          f"<J> {mb:.4e}); snapshot z = {z:.5f}, weight of {a.render_snap_b} {w:.4f}; <J> {J.mean(dtype=np.float64):.4e} "
          f"(<J_b>/<J_a> = {mb / ma:.4f})")
    return J, {"J_interp": 1, "render_snap_b": a.render_snap_b, "z_render_a": za, "z_render_b": zb, "J_weight_b": w}


def render_redshifts(dirs):
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "All", "All_*.hdf5")):
            s = int(os.path.basename(p)[4:7])
            try:
                with h5py.File(p, "r") as f:
                    out[s] = float(f["Header"].attrs["Redshift"])
            except Exception:
                pass
    return out


def inspect(a):
    snaps = sorted(int(os.path.basename(p)[5:8]) for p in glob.glob(os.path.join(a.snapdir, "snap_*.hdf5")))
    print(f"{a.snapdir}: {len(snaps)} snapshots; top level: {sorted(os.listdir(a.snapdir))[:12]}")
    rz = render_redshifts(a.renders) if a.renders else {}
    print(f"{'snap':>5} {'z':>9}" + (f" {'render':>7} {'z_render':>9} {'dz':>9}  J files" if rz else ""))
    for s in snaps:
        with h5py.File(os.path.join(a.snapdir, f"snap_{s:03d}.hdf5"), "r") as f:
            z = float(f["Header"].attrs["Redshift"])
        line = f"{s:5d} {z:9.5f}"
        if rz:
            r = min(rz, key=lambda k: abs(rz[k] - z))
            have = []
            if a.jdir:
                for tag, fs in (("H", (f"Jalpha_H_{r}.hdf5", f"Jalpha_AGN_H_{r}.hdf5")),
                                ("He", (f"Jalpha_He_{r}_{a.tr_tag}.hdf5", f"Jalpha_AGN_He_{r}_{a.tr_tag}.hdf5"))):
                    if all(os.path.exists(os.path.join(a.jdir, x)) for x in fs):
                        have.append(tag)
            line += f" {r:7d} {rz[r]:9.5f} {z - rz[r]:+9.2e}  {','.join(have) or '-'}"
        print(line)
    if a.snap is None:
        return
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    print(f"\nsnapshot {a.snap}: z = {S.z:.5f}, BoxSize {S.box}, h {S.h}, {S.n:,} gas cells, fields from {S.gas_path}")
    for k in ("Omega0", "OmegaBaryon", "OmegaLambda", "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s",
              "NumFilesPerSnapshot"):
        print(f"  {k}: {S.hd.get(k, '(absent)')}")
    print(f"  positions: {S.pos_name}; ElectronAbundance stored: {S.has_ne}"
          + ("" if S.has_ne else " (rebuilt as HII + HeII + 2 HeIII)"))
    for k in [S.pos_name] + [f for f in GAS_FIELDS if f != "Coordinates"]:
        d = S.g[k]
        print(f"  {k:18s} {str(d.shape):22s} {str(d.dtype):8s} chunks {d.chunks} compression {d.compression} "
              f"virtual {d.is_virtual}")
    for k in ("MultipleDomains", "TopNodeFactor", "NTask", "NumFilesPerSnapshot", "BITS_FOR_POSITIONS",
              "OUTPUT_COORDINATES_AS_INTEGERS"):
        if k in S.hd:
            print(f"  {k}: {S.hd[k]}")
    pa = dict(S.g[S.pos_name].attrs)
    if pa:
        print(f"  {S.pos_name} attributes: {pa}")
        if "to_cgs" in pa:
            fac = float(pa["to_cgs"]) / S.unit("UnitLength_in_cm", 3.085678e21)
            print(f"    to_cgs / UnitLength = {fac:.6e} code length per integer; BoxSize / 2^32 = {S.box / 2**32:.6e}")
    B = a.block_size
    print(f"\ncompactness of blocks of {B:,} consecutive cells (a Peano-Hilbert ordered file gives compact regions):")
    for i0 in (0, S.n // 3, 2 * S.n // 3):
        t0 = time.time()
        raw = S.g[S.pos_name][i0:i0 + B]
        dt = time.time() - t0
        x = S.read("Coordinates", i0, i0 + B)
        print(f"  {S.pos_name} rows {i0:,}+: raw min {raw.min(axis=0)}, max {raw.max(axis=0)}; "
              f"as positions {x.min():.4g}-{x.max():.4g} (BoxSize {S.box})")
        ext = []
        for ax in range(3):                                        # periodic extent: largest gap complement
            c = np.sort(x[:, ax])
            gaps = np.diff(np.concatenate([c, [c[0] + S.box]]))
            ext.append(S.box - gaps.max())
        side = (B * S.box**3 / S.n)**(1.0 / 3.0)
        print(f"  cells {i0:,}+: extent {ext[0] / 1e3:.1f} x {ext[1] / 1e3:.1f} x {ext[2] / 1e3:.1f} cMpc/h "
              f"(a cube of the same mean volume: {side / 1e3:.1f}); read {raw.nbytes / dt / 1e9:.2f} GB/s")
        # file order: steps between consecutive cells (minimum image); a jump starts a new compact piece
        dx = np.diff(x, axis=0)
        dx -= S.box * np.round(dx / S.box)
        step = np.sqrt((dx**2).sum(1))
        spacing = (S.box**3 / S.n)**(1.0 / 3.0)
        for jump in (10 * spacing, 1000.0):
            cut = np.flatnonzero(step > jump)
            lens = np.diff(np.concatenate([[0], cut + 1, [len(x)]]))
            print(f"    pieces split at steps > {jump / 1e3:.3f} cMpc/h: {len(lens):,}; cells per piece "
                  f"median {np.median(lens):.0f}, 90% {np.percentile(lens, 90):.0f}, max {lens.max():,}; "
                  f"cells in pieces >= 1e5: {lens[lens >= 100000].sum() / len(x):.3f}")
        print(f"    step percentiles 50/90/99%: {np.percentile(step, [50, 90, 99]).round(2).tolist()} ckpc/h "
              f"(mean cell spacing {spacing:.1f})")
        # what the test needs: are the cells in whole 320^3 pixels (compact patches), or scattered?
        V = S.read("Masses", i0, i0 + B) / S.read("Density", i0, i0 + B)
        for Ng in (40, 320):
            _, inv, _ = group(x, S.box, Ng)
            cover = np.bincount(inv, V) / (S.box / Ng)**3
            full = np.abs(cover - 1.0) < a.complete
            print(f"    {Ng}^3 pixels ({S.box / Ng / 1e3:.2f} cMpc/h): {len(cover):,} touched, {full.sum():,} complete "
                  f"(|gas volume / pixel - 1| < {a.complete}); cells in complete pixels {full[inv].mean():.3f}; "
                  f"cover quartiles {np.percentile(cover, [25, 50, 75]).round(3).tolist()}")


def group(pos, box, Ng):
    ix = np.floor(pos * (Ng / box)).astype(np.int64) % Ng
    key = (ix[:, 0] * Ng + ix[:, 1]) * Ng + ix[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    return uk, inv.ravel(), ix


ORDERS = ((0, 1, 2), (2, 1, 0))
GRIDS = ("V", "V_tb", "m", "m_T", "m_xHII", "m_fHeII")       # 640^3 sums of the cells, in render index order
GRIDS_SLIM = ("V", "V_tb")       # --grids slim: only what the result needs (the render checks need the full set)


class Phys:
    """constants and delta T_b of one line for a snapshot (tb_physics.py, the C-code physics)"""
    def __init__(self, S, line, spin_model=1):
        self.z, self.box, h = S.z, S.box, S.h
        om = float(S.hd.get("Omega0", 0.30964144)); ob = float(S.hd.get("OmegaBaryon", 0.048975))
        ol = float(S.hd.get("OmegaLambda", OMEGA_L_DEFAULT))
        self.UL, UM, self.UV = S.unit("UnitLength_in_cm", 3.085678e21), S.unit("UnitMass_in_g", 1.989e43), S.unit("UnitVelocity_in_cm_per_s", 1e5)
        self.rho_unit = UM * h**2 / self.UL**3                    # code density -> g cm^-3 comoving
        H0 = 100.0 * h * 1e5 / MPC
        self.rho_mean = ob * 3.0 * H0**2 / (8.0 * np.pi * G) / self.rho_unit
        self.line = line
        self.L = Line(line, self.z, h, om, ob, ol, spin_model=spin_model)

    def tb(self, rho, T, xHII, fII, fIII, J):
        nH = X_H * rho * self.rho_unit * (1.0 + self.z)**3 / M_H
        sig = rho * (1.0 - xHII) / self.rho_mean if self.line == "H" else (fII / Y_HE) * rho / self.rho_mean
        return self.L.tb(sig, nH, T, xHII, fII, fIII, J)[0]


def prep_cells(S, i0, i1):
    """the cells of rows i0:i1 with the render conventions: positions, m, rho, V, T, x_HII, f_HeII, f_HeIII"""
    F = {k: S.read(k, i0, i1) for k in GAS_FIELDS + ("ElectronAbundance",)}
    sf = F["StarFormationRate"] > 0
    mu = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC + 4.0 * HYDROGEN_MASSFRAC * F["ElectronAbundance"]) * PROTONMASS
    c = {"pos": F["Coordinates"], "m": F["Masses"], "rho": F["Density"],
         "T": GAMMA_MINUS1 * F["InternalEnergy"] * S.unit("UnitVelocity_in_cm_per_s", 1e5)**2 / BOLTZMANN * mu,
         "xHII": np.where(sf, 0.0, F["HII_Fraction"]), "fII": np.where(sf, 0.0, F["HeII_Fraction"]),
         "fIII": np.where(sf, 0.0, F["HeIII_Fraction"])}
    c["V"] = c["m"] / c["rho"]
    return c


def render_index(pos, box, N, order):
    ix = np.floor(pos * (N / box)).astype(np.int64) % N
    return ix[:, order[0]], ix[:, order[1]], ix[:, order[2]]


def choose_order(S, rows, Rd, box):
    """render axis order from the data: corr(log rho_cell, log render Density at the cell's pixel) for each order"""
    c = prep_cells(S, rows[0], rows[1])
    N = Rd.shape[0]
    res = []
    for o in ORDERS:
        i, j, k = render_index(c["pos"], box, N, o)
        rr = Rd[i, j, k].astype(np.float64)
        ok = rr > 0
        res.append((np.corrcoef(np.log(c["rho"][ok]), np.log(rr[ok]))[0, 1], o))
    err = 1.0 / np.sqrt(len(c["rho"]))                            # statistical error of a correlation coefficient
    for cc, o in res:
        print(f"  axis order render[i0,i1,i2] = coords {o}: corr(log rho_cell, log render Density) = {cc:.4f} (+- {err:.4f})")
    res.sort(reverse=True)
    # a single cell's density scatters by decades inside its pixel, so the right order gives a modest correlation
    # (0.2-0.7 on Lumina, but 0.047 for one block of rows at snapshot 131, vs 0.000 for the wrong order); it must be
    # significantly positive and stand out from the wrong one by many standard errors (no absolute floor)
    if not (res[0][0] > 10 * err and res[0][0] - res[1][0] > 20 * err):
        sys.exit("STOP: the render axis order is not clear from the data (see the correlations above)")
    return res[0][1]


def cube_origins(n_cube, side, seed, Nc):
    """n_cube distinct cubes of side^3 pixels of the Nc^3 grid (Nc = N/2, 320 for Lumina), aligned, no overlap"""
    if Nc % side:
        sys.exit(f"--cube-side must divide {Nc}")
    nl = Nc // side
    if n_cube > nl**3:
        sys.exit(f"--n-cube {n_cube} > {nl**3} lattice cubes")
    pick = np.random.default_rng(seed).choice(nl**3, size=n_cube, replace=False)
    return nl, {int(p): i for i, p in enumerate(pick)}


def scan(a):
    """one task: rows task::ntask blocks of the snapshot -> 640^3 sums (whole box) and the cells of the cubes"""
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    P = Phys(S, a.line, a.spin_model)
    box = S.box
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"scan_{a.line}_{a.snap:03d}_{a.task:03d}.hdf5")
    if not 0 <= a.task < a.ntask:
        sys.exit(f"ERROR: --task {a.task} outside 0..{a.ntask - 1} (--ntask {a.ntask})")
    edges = np.linspace(0, S.n, a.ntask + 1).astype(np.int64)
    r0, r1 = int(edges[a.task]), int(edges[a.task + 1])
    if os.path.exists(out):                       # reuse only a task file of the same run (audit 2026-11-19)
        with h5py.File(out, "r") as f:
            fa = dict(f.attrs)
        un = check_tag(fa, a, out, {"snap": a.snap, "rows": (r0, r1), "n_cube": a.n_cube, "cube_side": a.cube_side,
                                    "seed": a.seed, "grids": a.grids, "igm_cuts": list(a.igm_cuts or []), "ntask": a.ntask})
        print(f"{out} exists, same settings{' (not recorded, not checked: ' + ', '.join(un) + ')' if un else ''}: task done")
        return
    p = os.path.join(a.render, "Density", f"Density_{a.render_snap:03d}.hdf5")
    Rd = h5py.File(p if os.path.exists(p) else os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r")["Density"][()]
    N = Rd.shape[0]
    J, Jinfo = J_at(a, N, S.z)
    if J.shape != (N,) * 3:
        sys.exit(f"ERROR: J is {J.shape}, render {Rd.shape}")
    print(f"snapshot {a.snap} z = {S.z:.5f}, line {a.line}, task {a.task}/{a.ntask}: rows {r0:,}-{r1:,} of {S.n:,}")
    order = choose_order(S, (r0, min(r1, r0 + 5_000_000)), Rd, box)
    del Rd
    Nc = N // 2
    nl, cubes = cube_origins(a.n_cube, a.cube_side, a.seed, Nc)
    grids = list(GRIDS if a.grids == "full" else GRIDS_SLIM)
    igm = {f"V_tb_lt{c:g}": c for c in (a.igm_cuts or [])}          # IGM-only signal: cells with 1 + delta < cut
    grids += list(igm)
    G_ = {g: np.zeros(N**3, np.float64) for g in grids}
    part = out + ".part"
    fo = h5py.File(part, "w")
    fo.attrs.update({"snap": a.snap, "z": S.z, "line": a.line, "order": order, "rows": (r0, r1), "n_cube": a.n_cube,
                     "cube_side": a.cube_side, "seed": a.seed, "render_snap": a.render_snap, "grids": a.grids,
                     "igm_cuts": list(a.igm_cuts or []), "rho_mean": P.rho_mean, **Jinfo,
                     "spin_model": a.spin_model, "ntask": a.ntask,
                     "J_files": ";".join(os.path.basename(x) for x in a.J),
                     "J_b_files": ";".join(os.path.basename(x) for x in a.J_b)})
    cg = {}
    CF = ("pos", "m", "rho", "V", "T", "xHII", "fII", "fIII", "J", "tb")
    t0, sV, sVtb, ncell = time.time(), 0.0, 0.0, 0
    for i0 in range(r0, r1, a.chunk):
        i1 = min(r1, i0 + a.chunk)
        c = prep_cells(S, i0, i1)
        i, j, k = render_index(c["pos"], box, N, order)
        c["J"] = J[i, j, k].astype(np.float64)
        c["tb"] = P.tb(c["rho"], c["T"], c["xHII"], c["fII"], c["fIII"], c["J"])
        lin = (i * N + j) * N + k
        del i, j, k
        wts = {"V": lambda: c["V"], "V_tb": lambda: c["V"] * c["tb"], "m": lambda: c["m"], "m_T": lambda: c["m"] * c["T"],
               "m_xHII": lambda: c["m"] * c["xHII"], "m_fHeII": lambda: c["m"] * c["fII"]}
        dl = c["rho"] / P.rho_mean
        for g in grids:
            w = (c["V"] * c["tb"] * (dl < igm[g])) if g in igm else wts[g]()
            G_[g] += np.bincount(lin, w, minlength=N**3)
        sV += c["V"].sum(); sVtb += (c["V"] * c["tb"]).sum(); ncell += i1 - i0
        # cells of the cubes (cube lattice in coordinate axes)
        cl = np.floor(c["pos"] * (Nc / box)).astype(np.int64) % Nc // a.cube_side
        ul, inv = np.unique((cl[:, 0] * nl + cl[:, 1]) * nl + cl[:, 2], return_inverse=True)
        cidc = np.array([cubes.get(int(q), -1) for q in ul])[inv.ravel()]
        for q in np.unique(cidc[cidc >= 0]):
            s = cidc == q
            if q not in cg:
                cg[q] = fo.create_group(f"cube_{q:03d}")
                for f_ in CF:
                    shp = (0, 3) if f_ == "pos" else (0,)
                    cg[q].create_dataset(f_, shp, np.float64 if f_ == "pos" else np.float32,
                                         maxshape=(None,) + shp[1:], chunks=True)
            g_ = cg[q]
            n0 = g_["m"].shape[0]; n1 = n0 + int(s.sum())
            for f_ in CF:
                g_[f_].resize(n1, axis=0)
                g_[f_][n0:n1] = c[f_][s]
        if (i0 - r0) // a.chunk % 20 == 0:
            el = time.time() - t0
            print(f"  rows {i1 - r0:,}/{r1 - r0:,}: {el / 60:.1f} min, ~{el * (r1 - r0) / (i1 - r0) / 3600:.1f} h total", flush=True)
    for g in grids:
        dt = np.float64 if (g == "V_tb" and a.grids == "full") else np.float32      # IGM grids: float32
        fo.create_dataset(g, data=G_[g].reshape(N, N, N).astype(dt))
    fo.attrs.update({"sum_V": sV, "sum_V_tb": sVtb, "n_cells": ncell})
    fo.close()
    os.replace(part, out)
    print(f"done: {ncell:,} cells, <dT_b>_cell (these rows) = {sVtb / sV:+.5e} {P.L.unit}, {(time.time() - t0) / 3600:.2f} h")


def ladder(c, P, box, R_tb, order, N):
    """one cube: cell level vs the cells grouped into 320^3 ... 10240^3 pixels, and the production map"""
    names = [str(n) for n in levels(N)] + ["cell"]
    scales = (N // 2, N)
    V = c["V"]
    res, maps = {}, {S_: {} for S_ in scales}
    sinv = {S_: group(c["pos"], box, S_)[1] for S_ in scales}
    for lv in names:
        if lv == "cell":
            v = c["tb"]
        else:
            inv = group(c["pos"], box, int(lv))[1]
            Mp, Vp = np.bincount(inv, c["m"]), np.bincount(inv, V)
            avg = lambda q: np.bincount(inv, c["m"] * q) / Mp
            v = P.tb(Mp / Vp, avg(c["T"]), avg(c["xHII"]), avg(c["fII"]), avg(c["fIII"]), np.bincount(inv, V * c["J"]) / Vp)[inv]
        res[lv] = (V * v).sum() / V.sum()
        for S_ in scales:
            maps[S_][lv] = np.bincount(sinv[S_], V * v) / np.bincount(sinv[S_], V)
    # production map on the cube's 640^3 pixels (volume of the cube = whole pixels)
    uk, inv6, ix6 = group(c["pos"], box, N)
    first = np.zeros(len(uk), np.int64); first[inv6[::-1]] = np.arange(len(inv6))[::-1]
    pix = ix6[first]
    res["production"] = float(R_tb[pix[:, order[0]], pix[:, order[1]], pix[:, order[2]]].astype(np.float64).mean())
    return res, maps


def merge(a):
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    P = Phys(S, a.line, a.spin_model)
    box, z, unit = S.box, S.z, P.L.unit
    files = sorted(glob.glob(os.path.join(a.out, f"scan_{a.line}_{a.snap:03d}_*.hdf5")))
    files = [f for f in files if not f.endswith(".part")]
    print(f"snapshot {a.snap} z = {z:.5f}, line {a.line}: {len(files)} task files")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    order, ntask = tuple(int(x) for x in A["order"]), None
    if not files:
        sys.exit(f"ERROR: no task files scan_{a.line}_{a.snap:03d}_*.hdf5 in {a.out}")
    unchecked = set()
    same = ("ntask", "n_cube", "cube_side", "seed", "grids")       # must agree between the task files
    for fn in files:                                   # the cells' J (scan) and the reference J (here) must be the same
        with h5py.File(fn, "r") as f:
            fa = dict(f.attrs)
        unchecked |= set(check_tag(fa, a, fn, {"snap": a.snap, **{k: A[k] for k in same if k in A}}))
    if unchecked:
        print(f"  note: task files without {', '.join(sorted(unchecked))} (written before 2026-11-19): not checked")
    rows = sorted(tuple(int(x) for x in h5py.File(fn, "r").attrs["rows"]) for fn in files)
    covered = sum(r1 - r0 for r0, r1 in rows)
    gaps = [(rows[i][1], rows[i + 1][0]) for i in range(len(rows) - 1) if rows[i][1] != rows[i + 1][0]]
    ends_ok = rows[0][0] == 0 and rows[-1][1] == S.n
    print(f"  rows covered {covered:,} of {S.n:,}" + (f"; GAPS/OVERLAPS {gaps[:5]}" if gaps or not ends_ok or covered != S.n else ""))
    if gaps or not ends_ok or covered != S.n:          # audit 2026-11-19: was a warning, and the maps were still written
        msg = "not the whole snapshot exactly once (tasks missing, or task files of different --ntask)"
        if not a.allow_partial:
            sys.exit(f"ERROR: {msg}; rerun the missing tasks, or --allow-partial for a test")
        print(f"  WARNING (--allow-partial): {msg}: whole-box numbers and maps are partial")
    G_ = None
    sV = sVtb = 0.0
    for fn in files:
        with h5py.File(fn, "r") as f:
            if tuple(int(x) for x in f.attrs["order"]) != order:
                sys.exit(f"ERROR: axis order differs in {fn}")
            gs = [g for g in GRIDS if g in f] + sorted(k for k in f if k.startswith("V_tb_lt"))
            if G_ is None:
                G_ = {g: f[g][()].astype(np.float64) for g in gs}
            else:
                if set(gs) != set(G_):
                    sys.exit(f"ERROR: {fn} has grids {gs}, the first file {sorted(G_)}")
                for g in gs:
                    G_[g] += f[g][()]
            sV += float(f.attrs["sum_V"]); sVtb += float(f.attrs["sum_V_tb"])
    N = G_["V"].shape[0]
    vpix = (box / N)**3
    # production (render) inputs and map
    R = {}
    for k in ("Density", "Temperature", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction"):
        p = os.path.join(a.render, k, f"{k}_{a.render_snap:03d}.hdf5")
        R[k] = h5py.File(p if os.path.exists(p) else os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r")[k]
    J, _ = J_at(a, N, z)
    R_tb = np.empty((N,) * 3, np.float32)
    for i0 in range(0, N, 32):
        sl = slice(i0, i0 + 32)
        F = {k: R[k][sl].astype(np.float64) for k in R}
        R_tb[sl] = P.tb(F["Density"], F["Temperature"], F["HII_Fraction"], F["HeII_Fraction"], F["HeIII_Fraction"],
                        J[sl].astype(np.float64))
    # checks: gas volume per pixel (always); with --grids full also the cells deposited at 640^3 vs the render
    cov = G_["V"] / vpix
    print(f"\n  check, gas volume of the cells per {N}^3 pixel / pixel volume: total {G_['V'].sum() / box**3:.6f}, "
          f"median {np.median(cov):.4f}, 1-99% {np.percentile(cov, 1):.4f}-{np.percentile(cov, 99):.4f}, empty pixels {(G_['V'] == 0).sum():,}")
    if "m" in G_:
        Rd = R["Density"][()].astype(np.float64)
        mc = G_["m"] / vpix
        ok = (G_["m"] > 0) & (Rd > 0)
        print(f"  check, cells deposited at {N}^3 (nearest pixel) vs the render (cube-overlap deposit):")
        print(f"    total mass ratio {G_['m'].sum() / (Rd.sum() * vpix):.6f}; "
              f"corr(log rho) {np.corrcoef(np.log(mc[ok]), np.log(Rd[ok]))[0, 1]:.4f}, median ratio {np.median(Rd[ok] / mc[ok]):.4f}")
        del Rd
    else:
        print("  (--grids slim: no render comparison of mass, T and fractions)")
    for g, k in (("m_T", "Temperature"), ("m_xHII", "HII_Fraction"), ("m_fHeII", "HeII_Fraction")):
        if g not in G_:
            continue
        r = R[k][()].astype(np.float64)
        q = G_[g][ok] / G_["m"][ok]
        if k == "Temperature":
            print(f"    Temperature: median render/cells {np.median(r[ok] / q):.4f}, mass-weighted mean ratio "
                  f"{(r[ok] * G_['m'][ok]).sum() / G_[g][ok].sum():.5f}")
        else:
            print(f"    {k}: median |render - cells| {np.median(np.abs(r[ok] - q)):.2e}, mass-weighted means "
                  f"{(r[ok] * G_['m'][ok]).sum() / G_['m'][ok].sum():.5f} / {G_[g][ok].sum() / G_['m'][ok].sum():.5f}")
        del r
    # whole box: cell level vs production
    cellmap = np.where(G_["V"] > 0, G_["V_tb"] / np.maximum(G_["V"], 1e-300), R_tb).astype(np.float32)
    n_empty = int((G_["V"] == 0).sum())                # pixels with no cell centre: production value (recorded in the maps)
    if n_empty:
        print(f"  NOTE: {n_empty:,} pixels hold no cell centre; the cell-level map uses the production value there")
    prov = {"line": a.line, "spin_model": a.spin_model, "render_snap": a.render_snap, "snap_full": a.snap,
            "J_files": ";".join(os.path.basename(x) for x in a.J), "J_interp": 1 if a.J_b else 0,
            "n_empty_pixels_production": n_empty}
    if a.J_b:
        prov.update({"render_snap_b": a.render_snap_b, "J_b_files": ";".join(os.path.basename(x) for x in a.J_b),
                     "J_weight_b": (z - render_z(a.render, a.render_snap)) / (render_z(a.render, a.render_snap_b)
                                                                               - render_z(a.render, a.render_snap))})
    mean_cell = sVtb / sV
    mean_prod = float(R_tb.astype(np.float64).mean())
    print(f"\n  WHOLE BOX, <dT_b> [{unit}]: production ({N}^3 render inputs) {mean_prod:+.5e}, cell level {mean_cell:+.5e}; "
          f"production / cell = {mean_prod / mean_cell:.4f}")
    print(f"    (cell-level map at {N}^3: volume-weighted average of the cells in each pixel; its mean "
          f"{float(cellmap.astype(np.float64).mean()):+.5e})")
    kc, cnt, Pk, X = spectra([cellmap, R_tb], box / 1e3)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X[0] / np.sqrt(Pk[0] * Pk[1])
    sel = np.flatnonzero((cnt > 0) & (Pk[0] > 0))
    print("    P_production / P_cell and r(k), k in h/cMpc:")
    for i in sel:
        print(f"      k = {kc[i]:7.3f}: {Pk[1][i] / Pk[0][i]:.4f}  r = {r[i]:.4f}")
    igm_maps = {}
    for g in sorted((g for g in G_ if g.startswith("V_tb_lt")), key=lambda x: float(x[7:])):
        cut = float(g[7:])
        mi = np.where(G_["V"] > 0, G_[g] / np.maximum(G_["V"], 1e-300), 0.0).astype(np.float32)
        igm_maps[cut] = mi
        mean_i = G_[g].sum() / sV
        kc2, cnt2, Pk2, X2 = spectra([mi, R_tb, cellmap], box / 1e3)
        with np.errstate(invalid="ignore", divide="ignore"):
            rp = X2[0] / np.sqrt(Pk2[0] * Pk2[1]); rc = X2[1] / np.sqrt(Pk2[0] * Pk2[2])
        sel2 = np.flatnonzero((cnt2 > 0) & (Pk2[0] > 0))
        pick = sorted({int(sel2[np.argmin(abs(kc2[sel2] - qq))]) for qq in (0.03, 0.1, 0.3, 1.0, 3.0)})
        print(f"\n  WHOLE BOX, IGM only (cells with 1 + delta < {cut:g}): <dT_b> {mean_i:+.5e} {unit} = {mean_i / mean_cell:.4f} of the "
              f"cell level; production / IGM = {mean_prod / mean_i:.4f}")
        print("    k [h/cMpc]: P_IGM/P_cell, P_production/P_IGM, r(production, IGM), r(IGM, cell all):  " + "  ".join(
            f"{kc2[i]:.2f}: {Pk2[0][i] / Pk2[2][i]:.4f}, {Pk2[1][i] / Pk2[0][i]:.3f}, {rp[i]:.3f}, {rc[i]:.3f}" for i in pick))
    if a.save_map:
        with h5py.File(os.path.join(a.out, f"cellmap_{a.line}_{a.snap:03d}.hdf5"), "w") as f:
            f["dTb_cell"] = cellmap; f["dTb_production"] = R_tb
            for cut, mi in igm_maps.items():
                f[f"dTb_cell_igm_lt{cut:g}"] = mi
            f.attrs.update({"z": z, "unit": unit, "order": order, "box": box, **prov})
        # the cell-level map in the format of the production maps (RIN_NNN.hdf5: dataset "rin", render index order),
        # for the bispectrum, VID or PS codes that read those
        with h5py.File(os.path.join(a.out, f"RIN_cell_{a.line}_{a.snap:03d}.hdf5"), "w") as f:
            f["rin"] = cellmap
            f["rin"].attrs["content"] = np.bytes_(f"delta T_b{'^He' if a.line == 'He' else ''} [{unit}], real space; gas-cell level "
                                                  f"(volume-weighted mean of the Voronoi cells in each pixel), test_cells_Tb.py")
            f.attrs.update({"z": z, "box_ckpc_h": box, "order": order, **prov})
        for cut, mi in igm_maps.items():
            with h5py.File(os.path.join(a.out, f"RIN_cell_igm{cut:g}_{a.line}_{a.snap:03d}.hdf5"), "w") as f:
                f["rin"] = mi
                f["rin"].attrs["content"] = np.bytes_(f"delta T_b{'^He' if a.line == 'He' else ''} [{unit}], real space; gas-cell "
                                                      f"level, only cells with 1 + delta < {cut:g} (sum over them / pixel gas volume)")
                f.attrs.update({"z": z, "box_ckpc_h": box, "order": order, "igm_cut": cut, **prov})
    # cubes: the resolution ladder
    nc = int(A["n_cube"])
    names = [str(n) for n in levels(N)] + ["cell", "production"]
    scales = (N // 2, N)
    rows_ = []
    mapsS = {S_: {lv: [] for lv in names[:-1]} for S_ in scales}
    for q in range(nc):
        parts = []
        for fn in files:
            with h5py.File(fn, "r") as f:
                if f"cube_{q:03d}" in f:
                    parts.append({k: f[f"cube_{q:03d}/{k}"][()].astype(np.float64) for k in f[f"cube_{q:03d}"]})
        if not parts:
            continue
        c = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        res, maps = ladder(c, P, box, R_tb, order, N)
        rows_.append(res)
        for S_ in scales:
            for lv in names[:-1]:
                mapsS[S_][lv].append(maps[S_][lv])
    if rows_:
        print(f"\n  CUBES ({len(rows_)} x {int(A['cube_side'])}^3 of the {N // 2}^3 pixels): <dT_b> and ratio to the cell level "
              f"(jackknife over cubes)")
        tot = {lv: np.array([r_[lv] for r_ in rows_]) for lv in names}
        n = len(rows_)
        for lv in names:
            rat = tot[lv].sum() / tot["cell"].sum()
            jk = np.array([(tot[lv].sum() - tot[lv][i]) / (tot["cell"].sum() - tot["cell"][i]) for i in range(n)])
            err = np.sqrt((n - 1) / n * ((jk - jk.mean())**2).sum())
            print(f"    {lv:>10}: {tot[lv].mean():+.5e}  ratio {rat:.4f} +- {err:.4f}")
        for S_ in scales:
            ref = np.concatenate(mapsS[S_]["cell"])
            print(f"    map smoothed to {S_}^3 pixels ({len(ref):,}): variance ratio to cell level / correlation")
            print("      " + "  ".join(f"{lv}: {np.var(np.concatenate(mapsS[S_][lv])) / np.var(ref):.4f}/"
                                       f"{np.corrcoef(np.concatenate(mapsS[S_][lv]), ref)[0, 1]:.4f}" for lv in names[:-2]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("inspect", "scan", "merge"):
        q = sub.add_parser(name)
        q.add_argument("--snapdir", required=True)
        q.add_argument("--gas-file", help="file with the PartType0 fields (default: PartType0/PartType0_NNN.hdf5, else snap_NNN.hdf5)")
    q = sub.choices["inspect"]
    q.add_argument("--snap", type=int)
    q.add_argument("--renders", nargs="*", help="ren_640 directories, to match render snapshots by z")
    q.add_argument("--jdir", help="J_alpha directory, to say which matched snapshots have J files")
    q.add_argument("--tr-tag", default="tr_sfmask")
    q.add_argument("--block-size", type=int, default=40_000_000)
    q.add_argument("--complete", type=float, default=0.05)
    for name in ("scan", "merge"):
        q = sub.choices[name]
        q.add_argument("--snap", type=int, required=True)
        q.add_argument("--line", choices=["H", "He"], default="He")
        q.add_argument("--spin-model", type=int, choices=[0, 1], default=1,
                       help="21 cm spin model (tb_physics; 1 from 2026-11-15, 0 for results made before); He unaffected")
        q.add_argument("--J", action="append", required=True, help="640^3 J_alpha files of the line (stars, AGN), summed; 'none' (alone) = J = 0")
        q.add_argument("--render", required=True, help="ren_640 directory of the matching render snapshot")
        q.add_argument("--render-snap", type=int, required=True)
        q.add_argument("--max-dz", type=float, default=1e-4,
                       help="largest |z_snapshot - z_render| accepted for --render-snap (the pixel reference)")
        q.add_argument("--J-b", action="append", default=[],
                       help="J files of a second render (--render-snap-b, same --render directory): J linear in z "
                            "between the two renders, which must bracket the snapshot z")
        q.add_argument("--render-snap-b", type=int)
        q.add_argument("--out", required=True, help="directory of the task files")
    q = sub.choices["scan"]
    q.add_argument("--task", type=int, required=True)
    q.add_argument("--ntask", type=int, required=True)
    q.add_argument("--chunk", type=int, default=20_000_000, help="rows per read")
    q.add_argument("--n-cube", type=int, default=64)
    q.add_argument("--cube-side", type=int, default=16, help="cube side in pixels of the N/2 grid (320^3 for Lumina; divides it)")
    q.add_argument("--seed", type=int, default=12345)
    q.add_argument("--igm-cuts", type=float, nargs="*", default=[],
                   help="also whole-box IGM-only signal grids: cells with 1 + delta < each cut (1 GB per cut per task)")
    q.add_argument("--grids", choices=["full", "slim"], default="full",
                   help="full: 6 640^3 sums incl. the render checks (~7.3 GB per task); slim: volume and volume x dT_b (2.1 GB)")
    sub.choices["merge"].add_argument("--save-map", action="store_true",
                                      help="write the 640^3 cell-level and production maps (and RIN_cell_<line>_<snap>.hdf5)")
    sub.choices["merge"].add_argument("--allow-partial", action="store_true",
                                      help="merge although the task files do not cover the snapshot exactly once (tests only)")
    a = p.parse_args()
    {"inspect": inspect, "scan": scan, "merge": merge}[a.mode](a)


if __name__ == "__main__":
    main()
