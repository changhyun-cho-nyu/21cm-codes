#!/usr/bin/env python3
"""Inspect the X-ray / no-X-ray AREPO-RT runs (SonjaXrayPaper): run setup, render layout, snapshots.
Usage: python3 inspect_xray_runs.py [BASE] > xray_inspect.txt 2>&1   (login node; a few minutes on NFS)"""
import glob, os, re, sys, subprocess
import numpy as np, h5py
try:
    import hdf5plugin  # noqa: F401  (compressed datasets)
except ImportError:
    pass

BASE = sys.argv[1] if len(sys.argv) > 1 else "/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper"
RUNS = ["withoutXrays", "withXrays"]
KEYS = re.compile(r"box|omega|hubble|time(begin|max)|unit|xray|x_ray|rt_|photon|uvb|treecool|render|movie|"
                  r"lightcone|bh|blackhole|agn|esc|sn|wind|mingastemp|npix|pixel|grid|softening", re.I)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).rstrip()


def val(v, n=160):
    v = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def attrs(o, ind="    "):
    for k, v in o.attrs.items():
        print(f"{ind}@{k} = {val(v)}")


def stats(d):
    a = d[()] if d.size <= 2**26 else d[tuple(slice(0, s, max(1, s // 64)) for s in d.shape)]
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return "empty"
    return f"min {a.min():.4g}  mean {a.mean():.4g}  max {a.max():.4g}  zeros {np.mean(a == 0):.3f}"


def tree(f, with_stats=False, skip=()):
    def vis(name, o):
        if any(name.startswith(s) for s in skip):
            return
        if isinstance(o, h5py.Dataset):
            print(f"  {name}  {o.shape} {o.dtype} chunks={o.chunks} comp={o.compression}"
                  + (f"\n      {stats(o)}" if with_stats else ""))
            attrs(o, "      ")
        else:
            print(f"  {name}/")
    f.visititems(vis)


def group_dict(f, g):
    return {k: val(v, 400) for k, v in f[g].attrs.items()} if g in f else {}


def hdr_z(f):
    for g in ("Header", "header", "/"):
        if g in f:
            a = f[g].attrs
            if "Redshift" in a:
                return float(a["Redshift"])
            if "Time" in a:
                return 1 / float(a["Time"]) - 1
    return float("nan")


cfg = {}
for run in RUNS:
    R, O = f"{BASE}/{run}", f"{BASE}/{run}/output"
    print(f"\n{'=' * 100}\n RUN {run}\n{'=' * 100}")
    print(sh(f"head -40 {R}/job.sh; echo; cat {R}/WARNINGS 2>/dev/null | head -20"))
    print("\n-- param.txt-usedvalues (selected keys)")
    with open(f"{R}/param.txt-usedvalues") as fp:
        print("".join(l for l in fp if l.split() and KEYS.search(l.split()[0])), end="")
    print("-- sharedData/output_reionization.list: "
          + sh(f"wc -l < {R}/sharedData/output_reionization.list") + " lines\n"
          + sh(f"head -3 {R}/sharedData/output_reionization.list; echo ...; tail -3 {R}/sharedData/output_reionization.list"))
    for t in ("eor.txt", "sfr.txt", "photons_stars.txt", "photons_bhs.txt", "rad.txt", "blackholes.txt", "total_rad_energy.txt"):
        if not os.path.exists(f"{O}/{t}"):
            print(f"-- {t}: missing"); continue
        print(f"-- {t}: {sh(f'wc -l < {O}/{t}')} lines\n" + sh(f"head -4 {O}/{t}; echo ...; tail -2 {O}/{t}"))
    print("-- sizes\n" + sh(f"du -sh {O}/movie/renderings/render_309 {O}/snapdir_060 {O}/groups_060 2>&1; "
                            f"ls {O}/movie/rlc | head; ls {O}/movie/projections002 | head -3; df -h {BASE} | tail -1"))

    # ---------------- renders
    rdirs = sorted(glob.glob(f"{O}/movie/renderings/render_*"))
    print(f"\n-- renders: {len(rdirs)} directories")
    rd = f"{O}/movie/renderings/render_309"
    files = sorted(glob.glob(f"{rd}/render_309.*.hdf5"), key=lambda p: int(p.split(".")[-2]))
    print(f"   render_309: {len(files)} files")
    with h5py.File(files[0], "r") as f:
        print(f"   {files[0]} (top-level attrs, groups' attrs, datasets with stats):")
        attrs(f)
        for g in f:
            if isinstance(f[g], h5py.Group):
                print(f"  [{g}]")
                attrs(f[g])
        tree(f, with_stats=True)
        h0 = {g: group_dict(f, g) for g in f if isinstance(f[g], h5py.Group)}
    print("   per file: dataset shapes and header attrs that differ from file 0")
    for p in files:
        with h5py.File(p, "r") as f:
            shp = {}
            f.visititems(lambda n, o: shp.__setitem__(n, o.shape) if isinstance(o, h5py.Dataset) else None)
            diff = {g: {k: v for k, v in group_dict(f, g).items() if h0.get(g, {}).get(k) != v}
                    for g in f if isinstance(f[g], h5py.Group)}
            print(f"   {os.path.basename(p)}: {sorted(set(shp.values()))}  diff={ {g: d for g, d in diff.items() if d} }")
    if True:  # both runs: are the render times the same?
        with open(f"render_z_{run}.csv", "w") as out:
            out.write("render,z\n")
            for d in rdirs:
                n = int(d.split("_")[-1])
                try:
                    with h5py.File(f"{d}/render_{n:03d}.0.hdf5", "r") as f:
                        out.write(f"{n},{hdr_z(f):.5f}\n")
                except OSError as e:
                    out.write(f"{n},nan\n")
        print(f"   render redshifts -> render_z_{run}.csv\n" + sh(f"sed -n '2,4p;100p;200p;300p;$p' render_z_{run}.csv"))

    # ---------------- snapshots
    for s in (0, 60, 116):
        fs = sorted(glob.glob(f"{O}/snapdir_{s:03d}/*.hdf5"))
        if not fs:
            print(f"-- snapdir_{s:03d}: no hdf5 files: {sh(f'ls {O}/snapdir_{s:03d} | head')}")
            continue
        with h5py.File(fs[0], "r") as f:
            print(f"\n-- snapdir_{s:03d}: {len(fs)} files, {fs[0]}")
            attrs(f["Header"])
            if s == 60:
                tree(f, with_stats=True, skip=("Config", "Parameters", "Header"))
            cfg[run] = (group_dict(f, "Config"), group_dict(f, "Parameters"))

print(f"\n{'=' * 100}\n Config (compile flags), {RUNS[1]}\n{'=' * 100}")
for k, v in sorted(cfg.get(RUNS[1], ({}, {}))[0].items()):
    print(f"  {k} = {v}")
for i, name in enumerate(("Config", "Parameters")):
    a, b = (cfg.get(r, ({}, {}))[i] for r in RUNS)
    print(f"\n-- {name}: differences {RUNS[0]} -> {RUNS[1]}")
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            print(f"  {k}: {a.get(k)} -> {b.get(k)}")
print("\n-- param.txt-usedvalues differences\n"
      + sh(f"diff {BASE}/{RUNS[0]}/param.txt-usedvalues {BASE}/{RUNS[1]}/param.txt-usedvalues"))
