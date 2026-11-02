"""Central registry for the paper's figures.

PURPOSE
-------
Every figure goes through :func:`save_figure`, so that the file names the scripts
write are exactly the names ``main.tex`` includes.  :func:`save_figure`

  * writes a vector PDF for the LaTeX build and a PNG for on-screen viewing, side by side;
  * uses one stable name per figure, declared here and nowhere else;
  * embeds the generating script and the statement the figure supports in the PDF
    metadata, so a figure in the paper can always be traced back to its source.

NAMING
------
Names describe the CONTENT, not the position in the experiment list, so a file name
does not change meaning if experiments are reordered or moved to the supplementary
material.

  main text     fig_<topic>.pdf
  supplementary figS_<topic>.pdf

Do not rename an existing entry; add a new one and retire the old.

USAGE
-----
    from soft_partition_paper.figures import save_figure
    save_figure(fig, 'vmd_agreement', save_dir)

To print the LaTeX include lines, or to check that ``figs/`` matches the registry:

    python experiments/soft_partition_paper/figures.py --latex
    python experiments/soft_partition_paper/figures.py --check path/to/paper/figs
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)              # experiments/
_ROOT = os.path.dirname(_EXP)              # repository root
for _p in (_ROOT, _EXP):                   # so the module runs from any directory
    if _p not in sys.path:
        sys.path.insert(0, _p)
from typing import Dict

__all__ = ["Figure", "FIGURES", "save_figure", "figure_path", "latex_lines"]


@dataclass(frozen=True)
class Figure:
    r"""One figure of the paper.

    Attributes
    ----------
    stem      : file name without extension; the single source of truth.
    supports  : the numbered statement(s) the figure is evidence for.
    script    : the script that generates it.
    width     : 'column' (\columnwidth, ~3.5 in) or 'text' (\textwidth, ~7.16 in,
                i.e. a figure* spanning both columns).  Must match the figsize the
                script uses, or the figure will be rescaled and its fonts will no
                longer match the body text.
    supplementary : True for figures that live outside the main text.
    """
    stem: str
    supports: str
    script: str
    width: str = "column"
    supplementary: bool = False

    @property
    def filename(self) -> str:
        return f"{self.stem}.pdf"


FIGURES: Dict[str, Figure] = {
    "vmd_agreement": Figure(
        stem="fig_vmd_agreement",
        supports="Rem. VI.5 -- the VMD ADMM fixed point is the inverse-square "
                 "allocation at every alpha; the Lorentzian family meets it only as alpha -> inf",
        script="exp_a.py",
        width="text",
    ),
    "closed_form": Figure(
        stem="fig_closed_form",
        supports="Thm. IV.3 -- the closed form is the minimiser of the regularised "
                 "objective, unique where P > 0",
        script="exp_h.py",
        width="text",
        supplementary=True,
    ),
    "lorentzian_limits": Figure(
        stem="fig_lorentzian_limits",
        supports="Cor. IV.9, Rem. IV.10 -- the two limits of the Lorentzian family, "
                 "and its distance from the Voronoi mask",
        script="exp_c.py",
        width="text",
    ),
    "mmse_shrinkage": Figure(
        stem="fig_mmse_shrinkage",
        supports="Thm. V.2, Rem. V.4 -- MMSE optimality in the low-noise limit and "
                 "the finite-noise shrinkage factor",
        script="exp_d.py",
        width="text",
    ),
    "robustness": Figure(
        stem="fig_robustness",
        supports="Exp. C' -- spectral shape, centre error and separation; drawn by "
                 "plot_robustness_3panel.py from exp_robustness.py and the sweep .npz",
        script="plot_robustness_3panel.py",
        width="text",
    ),
    "alpha_estimation": Figure(
        stem="fig_alpha_estimation",
        supports="Cor. V.5 -- accuracy of the FWHM estimator against bins per FWHM",
        script="exp_e.py",
        width="text",
        supplementary=True,
    ),
    "synthetic_suite": Figure(
        stem="fig_synthetic_suite",
        supports="hard and soft extraction on the four signals of synthetic_suite.py",
        script="exp_d_synthetic.py",
        width="text",
    ),
    "runtime": Figure(
        stem="fig_runtime",
        supports="Table II, Alg. 1 -- stage-separated cost against STVMD",
        script="exp_g.py",
        width="text",
        supplementary=True,
    ),
    # -- supplementary -------------------------------------------------------
    "wiener_identity": Figure(
        stem="figS_wiener_identity",
        supports="Rem. IV.5 -- implementation consistency between two evaluations of "
                 "the same expression; a unit test, not evidence",
        script="exp_b.py",
        width="column",
        supplementary=True,
    ),
}


def figure_path(key: str, save_dir: str, ext: str = "pdf") -> str:
    if key not in FIGURES:
        raise KeyError(f"unknown figure key {key!r}; known: {sorted(FIGURES)}")
    return os.path.join(save_dir, f"{FIGURES[key].stem}.{ext}")


def save_figure(fig, key: str, save_dir: str, *, dpi: int = 300,
                also_png: bool = True, close: bool = True) -> str:
    """Write `fig` as PDF (for LaTeX) and PNG (for viewing).  Returns the PDF path."""
    import matplotlib.pyplot as plt

    if key not in FIGURES:
        raise KeyError(f"unknown figure key {key!r}; add it to FIGURES first")
    spec = FIGURES[key]
    os.makedirs(save_dir, exist_ok=True)

    pdf_path = figure_path(key, save_dir, "pdf")
    # Embed fonts as TrueType (Type 42), never as Type 3, whatever the calling script set.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf_path, bbox_inches="tight",
                    metadata={"Title": spec.stem,
                              "Subject": spec.supports,
                              "Creator": spec.script})
    written = [pdf_path]
    if also_png:
        png_path = figure_path(key, save_dir, "png")
        fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
        written.append(png_path)
    if close:
        plt.close(fig)

    for p in written:
        print(f"Saved {p}")
    return pdf_path


def latex_lines() -> str:
    """The includegraphics lines the paper uses."""
    out = []
    for key, f in FIGURES.items():
        w = r"\columnwidth" if f.width == "column" else r"\textwidth"
        tag = "  % supplementary" if f.supplementary else ""
        out.append(rf"\includegraphics[width={w}]{{figure/{f.filename}}}{tag}")
    return "\n".join(out)


def check(figs_dir: str) -> int:
    """Report missing and unknown files in a figures directory."""
    present = {f for f in os.listdir(figs_dir)} if os.path.isdir(figs_dir) else set()
    expected = {f.filename for f in FIGURES.values()}

    missing = sorted(expected - present)
    unknown = sorted(p for p in present if p.endswith(".pdf") and p not in expected)

    print(f"figures directory: {figs_dir}")
    print(f"  expected {len(expected)}, present {len(present & expected)}")
    for m in missing:
        print(f"  MISSING  {m}")
    for u in unknown:
        print(f"  UNKNOWN  {u}  (not in the registry)")
    if not (missing or unknown):
        print("  ok")
    return 1 if missing else 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--latex", action="store_true",
                    help="print the includegraphics lines for main.tex")
    ap.add_argument("--list", action="store_true", help="print the registry")
    ap.add_argument("--check", metavar="FIGS_DIR",
                    help="check a figures directory against the registry")
    args = ap.parse_args()

    if args.latex:
        print(latex_lines())
    elif args.check:
        sys.exit(check(args.check))
    else:
        w = max(len(f.stem) for f in FIGURES.values())
        for key, f in FIGURES.items():
            mark = "S" if f.supplementary else " "
            print(f"{mark} {f.stem:<{w}}  <- {f.script:<18}  {f.supports}")
