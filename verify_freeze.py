"""Check every frozen copy in paper/upgrade_freeze/ against the SHA-256 hash recorded when it was frozen.
python3 verify_freeze.py      (from the repository root; prints one line per hash file and a summary)

The hash files use three path conventions, from different stages of the project:
  a path relative to paper/upgrade_freeze/ (or to a code folder inside it);
  a path relative to the repository root (paper/upgrade_freeze/...);
  "HYPOTHESES.md", the file hashed just before it was copied to HYPOTHESES_<stamp>.md.
"""
import glob
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FREEZE = os.path.join(ROOT, "paper", "upgrade_freeze")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidates(hash_file, name):
    stem = os.path.basename(hash_file)[: -len(".sha256")]
    code_dir = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem
    out = [os.path.join(FREEZE, name), os.path.join(ROOT, name), os.path.join(FREEZE, code_dir, name)]
    if os.path.basename(name) == "HYPOTHESES.md":
        out.append(os.path.join(FREEZE, stem + ".md"))
    return out


ok = bad = 0
for hf in sorted(glob.glob(os.path.join(FREEZE, "*.sha256"))):
    good, total, missing = 0, 0, []
    for line in open(hf):
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        total += 1
        if any(os.path.exists(c) and sha(c) == want for c in candidates(hf, name)):
            good += 1
        else:
            missing.append(name)
    status = "OK" if good == total else "MISMATCH " + ", ".join(missing)
    print(f"{status:10s} {os.path.basename(hf)} ({good}/{total} files)")
    ok, bad = ok + (good == total), bad + (good != total)
print(f"\n{ok} hash files verified, {bad} not verified")
sys.exit(1 if bad else 0)
