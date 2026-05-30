# Problems Log

## 2026-05-22 - Right-Stack Placement + Verify V2 Mismatch

### Observed
- Right-stack placement could target the wrong XY (not clearly on top of the existing right cube).
- At effective stack height 3 (after manual movement), verify reported `placed_mismatch_out_of_margin`.
- `verify_v2` then surfaced a state consistent with missing top (`top=empty` in observed behavior).

### Current Hypotheses
- Verify margins were too strict for real-world measurement jitter at top-stack levels.
- Expected-vs-measured reference may still be sensitive to offset alignment (`expected` vs `expected_eval` interpretation).
- Manual scene movement can desync inferred stack state vs live measured geometry.

### Minimal Fix Applied (today)
- Relaxed verify-v2 mismatch scoring margins (verify-only):
  - `QARM_PLACE_VERIFY_V2_MISMATCH_RELAX_XY_M` default `0.012`
  - `QARM_PLACE_VERIFY_V2_MISMATCH_RELAX_Z_M` default `0.006`
- Scope:
  - Applied only to verify-v2 geometry scoring (`placed_mismatch_out_of_margin` sensitivity).
  - No tracker, centering, or startup-hydrate behavior changes in this patch.

### Tomorrow Test Plan
- Run same right-stack scenario with the new mismatch relax defaults.
- Compare logs for:
  - `expected`, `expected_eval`, `measured`
  - `effective_xy_margin_m`, `effective_z_margin_m`
  - final `status` from verify-v2
- Validate whether true top placements are accepted instead of false mismatch rejection.
- If false mismatches remain, test offset alignment next (`QARM_PLACE_VERIFY_V2_EXPECTED_EVAL_USE_OFFSETS`) with controlled A/B runs.
