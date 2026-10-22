"""Load the authors' own code at pinned commits, at run time.

Nothing of theirs is copied into this repository. holis (AGPL-3.0) is one
Jupyter notebook; I read its JSON and exec its code cells into a fresh
module namespace. residuelisp (MIT) is a Python package; I unpack its
source tree from the pinned commit and put src/ on sys.path.

Sources, in order:
  1. a local clone named by $HOLIS_CLONE / $RESIDUELISP_CLONE (git show / git archive),
  2. the cache in repro/holis/.upstream/ (gitignored),
  3. a download from GitHub at the pinned commit.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import types
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".upstream")

HOLIS_REPO = "eilene-ftf/holis"
HOLIS_COMMIT = "db208d782863febfb50ec46b25a3700369f5cfea"
RESIDUELISP_REPO = "hanleyc01/residuelisp"
RESIDUELISP_COMMIT = "09864c74f1a35680a0008e59d7ab2571269a986f"


def _fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def holis_notebook_bytes():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"holis-{HOLIS_COMMIT[:7]}.ipynb")
    if not os.path.exists(path):
        clone = os.environ.get("HOLIS_CLONE")
        if clone:
            data = subprocess.check_output(
                ["git", "-C", clone, "show", f"{HOLIS_COMMIT}:holis.ipynb"])
        else:
            data = _fetch(f"https://raw.githubusercontent.com/{HOLIS_REPO}/"
                          f"{HOLIS_COMMIT}/holis.ipynb")
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        return f.read()


def holis_provenance():
    data = holis_notebook_bytes()
    return {"repo": f"https://github.com/{HOLIS_REPO}", "commit": HOLIS_COMMIT,
            "file": "holis.ipynb", "sha256": hashlib.sha256(data).hexdigest()}


def holis_cells():
    nb = json.loads(holis_notebook_bytes())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def load_holis():
    """Exec the notebook's code cells into a module and return it.

    Code cells, in order: 0 HRR classes, 1 vsa_env, 2 lispy (builds a
    global_env at dim 512), 3 the fizzbuzz definitions (I take only the
    program string; its run() call is suppressed), 4 repl() (skipped).
    """
    cells = holis_cells()
    assert len(cells) == 5, f"unexpected notebook layout: {len(cells)} code cells"
    mod = types.ModuleType("holis")
    ns = mod.__dict__
    for src in cells[:3]:
        exec(compile(src, "holis.ipynb", "exec"), ns)
    # cell 3 defines lisp_fizzbuzz_program and calls run(); keep the string only
    fb = {"run": lambda txt: None}
    exec(compile(cells[3], "holis.ipynb", "exec"), fb)
    mod.lisp_fizzbuzz_program = fb["lisp_fizzbuzz_program"]
    return mod


def residuelisp_src():
    """Return the path of residuelisp's src/ at the pinned commit."""
    root = os.path.join(CACHE, f"residuelisp-{RESIDUELISP_COMMIT[:7]}")
    src = os.path.join(root, "src")
    if not os.path.isdir(src):
        os.makedirs(CACHE, exist_ok=True)
        clone = os.environ.get("RESIDUELISP_CLONE")
        if clone:
            data = subprocess.check_output(
                ["git", "-C", clone, "archive", "--format=tar", RESIDUELISP_COMMIT])
            tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
            tf.extractall(root, filter="data")
        else:
            data = _fetch(f"https://codeload.github.com/{RESIDUELISP_REPO}/"
                          f"tar.gz/{RESIDUELISP_COMMIT}")
            tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
            prefix = tf.getnames()[0].split("/")[0]
            for m in tf.getmembers():
                m.name = m.name[len(prefix) + 1:] if m.name != prefix else "."
            tf.extractall(root, filter="data")
    return src


def residuelisp_provenance():
    return {"repo": f"https://github.com/{RESIDUELISP_REPO}",
            "commit": RESIDUELISP_COMMIT}


def load_residuelisp():
    """Import residuelisp's packages (vsa, syntax, language) and return them."""
    src = residuelisp_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    import language  # noqa: F401
    import language.encoding as encoding
    import language.interpreter as interpreter
    import syntax
    import vsa
    return types.SimpleNamespace(vsa=vsa, syntax=syntax, encoding=encoding,
                                 interpreter=interpreter)
