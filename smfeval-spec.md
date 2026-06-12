# smfeval — tool specification

Role: the living surface. Verdict machine for probabilistic SLAM evaluation.
Everything a stranger needs to score their own filter lives here. Nothing
paper-specific does.

## Scope discipline

- Scoring core stays at numpy + scipy. No experiment harness, no dataset
  adapters, no container recipes, no plotting stacks in the package
  dependencies. If a feature would add a dependency, it belongs in a
  separate repo or an optional extra.
- smfeval never absorbs a research pipeline. slam_benchmark consumes
  smfeval, not the reverse.

## README first screen (the thirty-second test)

The first screen must show, in order:

1. One-line pitch: score the belief, not just the mean.
2. Install: `pip install smfeval` with a note that the only dependencies
   are numpy and scipy.
3. A single command and its output, ending in a verdict line. Target shape:

   ```
   $ smfeval nees estimate.square reference.tum
   median NEES 5.63e3   (calibrated: 2.37)
   covariance scale gap k = 5.63e3, ~75x too tight per axis
   90% coverage: 0.000  (calibrated: 0.900)
   ```

   The verdict line is the product. k and 2.37 are interpretable without
   reading the paper; keep them as defaults.
4. The no-reference mode, stated as the headline capability:
   "No ground truth? Run two filters and score them against each other;
   an elevated pairwise NEES certifies overconfidence with no reference
   consulted." Command: `smfeval pair a.square b.square`. This is the
   structural advantage over evo; lead with it.

Taxonomy of scoring rules, SQUARE format details, and theory go below the
fold or in docs, never on the first screen.

## CLI

- evo-style verbs, not library-first: `smfeval nees`, `smfeval pair`,
  `smfeval score` (or current equivalents). A new user should never need
  to write Python to get a verdict.
- Output is the verdict block above; machine-readable (json) behind a flag.

## On-ramp: input formats

- SQUARE is the native format; the spec is documented here (done).
- Escape hatch so SQUARE adoption is not a precondition for trying the
  tool: accept TUM-format poses plus a separate covariance file, or
  pose + flattened lower-triangle in extra columns. Document both.
- A user with a filter that prints covariance anywhere must be able to
  get a verdict in under an hour.

## Exporters (canonical home: here)

- One directory per filter under `docs/exporters/` (or `exporters/`),
  each containing:
  - the diff,
  - the upstream commit it applies to,
  - a validation run: smfeval output on one named public sequence.
- The four audited exporters (Fast-LIO2, Faster-LIO, Point-LIO,
  I2EKF-LO) move here from slam_benchmark before the paper release.
  "Here is the four-line diff that makes FAST-LIO2 write its belief" is
  the single most copyable artifact; name it as such.
- Community contributions:
  - PR template requires the diff, the upstream commit, and the
    validation run attached.
  - Mechanical checks: covariance SPD, plausible magnitude, NEES not
    degenerate-zero.
  - Status split in docs: `verified` (audited against source and
    empirical error scatter, the Sec. V.d standard) vs `contributed`.
    A wrong export produces a wrong verdict attributed to the tool;
    the gate protects the audit standard without making you the
    bottleneck.
- Exporter count (N filters, M community-contributed) is the adoption
  metric for DDSA reporting; design the directory so it is countable.

## Notebook

- Colab badge in the README. Notebook loads one Oxford Spires sequence
  and reproduces the Fig. 1 verdict (median NEES, k, coverage) end to
  end. Use the existing Jupytext `# %%` workflow; `.py` as source of
  truth, notebook generated.

## Citation and metadata

- `CITATION.cff` pointing at the tool (software citation; later JOSS or
  the smfeval section of the paper, decide then). Keeps the tool citation
  stream separable from the paper's.
- Issue template asks for the SQUARE file (or input files) up front;
  support burden of a measurement tool is mostly malformed input.
- README links slam_benchmark as "the audit that motivated this tool"
  (credibility import) with its DOI once minted.

## Release tied to the paper freeze

- Before slam_benchmark freezes: exporters migrated in, then cut the
  paper-matching release and publish to PyPI. The tag is what
  slam_benchmark pins.
- Versioned releases thereafter as normal; the artifact never constrains
  the tool again.
- Later, optional: Zenodo webhook on this repo too, so tool releases
  carry DOIs and the software is independently citable. Not needed for
  the paper.
