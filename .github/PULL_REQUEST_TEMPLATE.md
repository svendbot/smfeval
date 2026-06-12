# Pull request

<!-- For code changes, describe the change and how you tested it. -->

## Exporter contribution checklist

<!-- Delete this section if the PR does not add or change an exporter.
     New exporters land with status `contributed`; promotion to `verified`
     requires the source + error-scatter audit (see exporters/README.md). -->

- [ ] `exporters/<filter>/belief-publisher.patch` — the diff that makes the
      filter publish its belief
- [ ] `exporters/<filter>/UPSTREAM` — the upstream repo and commit the patch
      applies to, as `URL @ sha`
- [ ] `exporters/<filter>/bag_to_square.py` (or equivalent) — the recorded
      topic → SQUARE converter
- [ ] `exporters/<filter>/VALIDATION.md` — a validation run on a **named
      public sequence**: the exact `smfeval nees` (or `score`) command and
      its verbatim verdict block
- [ ] Output of `smfeval validate --strict your.SQUARE` pasted below
      (covariance SPD, plausible magnitude, not degenerate-zero)

```
# smfeval validate --strict output here
```

Declared conventions (must match what the filter actually publishes):

| Field | Value |
|---|---|
| `BODY_FRAME` | |
| `TANGENT_CONVENTION` | |
| `TANGENT_ORDER` | |
| `GAUGE` | |
