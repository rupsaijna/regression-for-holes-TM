"""explanations_of_runs.py — narrative documentation for the teacher-guided
RegressionTM distillation arc (experiments E8-E11).

This is a DOCUMENTATION module, not an experiment. It collects the detailed
explanations built up while running and analysing the distillation experiments:
the core mechanism, the catalogue of teacher-correction modes, worked numerical
examples, per-experiment findings, and the cross-cutting analyses (vs no-teacher,
vs the teacher M20, vs a full-data model, and the teacher-edge correlation).

Run it to navigate:
    python explanations_of_runs.py             # list sections
    python explanations_of_runs.py 5           # print section 5
    python explanations_of_runs.py all         # print everything

Nothing here imports tmu, so it runs anywhere.
"""

# ===========================================================================
SECTIONS = []
def _s(title, body): SECTIONS.append((title, body.strip("\n")))


# ---------------------------------------------------------------------------
_s("0. Run setup / environment", '''
- tmu cannot build on Windows Python 3.14 (no C compiler). All experiments run
  in WSL Ubuntu (Python 3.10), venv at ~/regexp-venv. Repo is reached from WSL
  at /mnt/c/Users/Trshant/Rupsa/TMExperiments/RegressionExperiment.
- A LOCAL editable copy of tmu is vendored at tmu/ in the repo, so the patched
  regressors (teacher-guided fit) are what run. Each prints a one-line
  "[tmu LOCAL BUILD] ..." sentinel on import to confirm which file loaded.
- Pure-analysis scripts (compare_references.py, correlate_teacher_edge.py,
  explanations_of_runs.py) import only numpy/matplotlib/config and run on Windows.
- Shared config (config.py): TM_NUM_CLAUSES=1000, TM_T=5000, TM_S=2.75,
  TM_MAX_BITS_PER_FEATURE=10, RANDOM_SEED=42. Experiments use 6 seeds (42..47),
  100 epochs, f=0.75, p_teacher=0.5 unless noted.
- VERSIONING RULE: never edit a shared library file in place; copy it to the next
  version (vanilla_regressor_v2/_v3/_v4.py) so every prior experiment stays
  reproducible against the exact code it ran on. Same class name (TMRegressor) so
  the new file is a drop-in import.
''')


# ---------------------------------------------------------------------------
_s("1. Project & masked-distillation setup", '''
Goal: teacher-guided knowledge distillation between two Regression Tsetlin
Machines, and whether/where it helps.

Per dataset, with a fixed train/test split:
  - nested_structured(Xs, [0.2, 0.3], seed) carves NESTED structured holes in the
    TRAIN set (cumulative k-NN neighbourhoods). It returns rem_t (20% removed) and
    rem_s (30% removed) with rem_t subset of rem_s.
  - TEACHER  M20 : trained on kept_t = all - rem_t   (80% of train, the 20% hole).
  - STUDENT (no-teacher baseline) M30u / "unguided" : trained on
                   kept_s = all - rem_s   (70% of train, the 30% hole).
  - GUIDED students: trained on the SAME 30%-hole data as M30u, but with the frozen
    teacher's per-sample correction applied during fit. The ONLY difference between
    unguided and a guided run is teacher on/off (same data, same seed).
  - FULL ceiling M00 : trained on ALL train rows (0% hole). Added in E11/full_baseline.

Because the holes are nested, the student's data is a strict subset of the
teacher's. Everything is evaluated on the SAME held-out TEST set (never masked).

Core metric:  impr = RMSE(unguided) - RMSE(guided).   impr > 0 => guidance beat
no-teacher. Paired by seed; t-stats over the 6 seeds (|t| >~ 2.57 = sig at 0.05).
''')


# ---------------------------------------------------------------------------
_s("2. The teacher-guided fit mechanism (the core math)", '''
Everything happens in the ENCODED label space [0, T], T=5000. The target y is
encoded as encoded_Y in [0,T]; the model prediction is
    pred_y = clip( sum(weights * clause_outputs), 0, T ).

Per training sample e, two error quantities:
    prediction_error = pred_y - encoded_Y[e]        # student's error
    pe_teacher       = teacher_pred[e] - encoded_Y[e]   # frozen teacher's error
(teacher_pred is PRECOMPUTED once via teacher_encoded_predictions(), so the
teacher's clause-bank calls never interleave with the student's updates -- the
two share C buffers, which would otherwise make the signal order-dependent.)

Two things derive from prediction_error:
  (a) WHICH feedback type fires -- the SIGN:
        error > 0 (predicted too HIGH) -> Type-II feedback (erode -> lowers pred)
        error < 0 (predicted too LOW)  -> Type-I  feedback (build  -> raises pred)
  (b) HOW HARD it updates -- the MAGNITUDE, as a per-clause probability:
        update_p = (magnitude / T) ** 2

The teacher fires only when its per-sample coin-flip succeeds (prob p_teacher,
drawn from a DEDICATED teacher_rng so the student's own RNG stream is untouched;
off == baseline bit-for-bit). When it fires, it adjusts the sample's update via
one of the modes in section 5. When it does NOT fire (~half the samples at p=0.5),
the sample is a plain student update.

Same-side blend (the original mechanism): if student and teacher errors share a
sign, move the error toward the teacher:
    prediction_error <- (1 - f) * prediction_error + f * pe_teacher
With f=0.75 the update is pulled 3/4 of the way to the teacher. Opposite-side
handling is what E9-E11 explore.
''')


# ---------------------------------------------------------------------------
_s("3. Feedback types: Type-I vs Type-II", '''
Tsetlin Machine feedback (regression variant):
  Type-I  feedback : reinforces/INCLUDES literals so clauses recognise the pattern;
                     net effect on a regressor = pushes the prediction UP. Fired when
                     the model under-predicts (prediction_error < 0). Pairs with
                     weight_bank.increment().
  Type-II feedback : erodes/EXCLUDES literals (combats false positives); net effect =
                     pushes the prediction DOWN. Fired when the model over-predicts
                     (prediction_error > 0). Pairs with weight_bank.decrement().

Crucial asymmetry exploited later: Type-II only ever LOWERS a prediction, and
predictions are floored at 0, so a stream of Type-II updates is self-limiting --
it cannot run away. Type-I RAISES, and if mis-targeted onto an already-too-high
prediction it can compound without an upper bound (the E9 runaway, section 8).

"force Type-II" therefore means: on a chosen subset of samples, apply Type-II
REGARDLESS of the error sign. It does NOT mean Type-I is abolished -- see section 5.
''')


# ---------------------------------------------------------------------------
_s("4. Versioned regressors v1-v4", '''
tmu/tmu/models/regression/:
  vanilla_regressor.py     (v1) : stock + the original teacher params
        (teacher_pred_encoded, f, f_opposite, p_teacher, teacher_p_sample,
         teacher_rng, global_y_min/max). Opposite-side handled by an f_opposite
         blend; f_opposite kept 0.0 in every run => teacher IGNORED on opposite side.
  vanilla_regressor_v2.py  (v2, E9) : adds opposite_mode in {None, force_ii,
         teacher_sign}. Overrides the feedback TYPE on opposite-side samples.
  vanilla_regressor_v3.py  (v3, E10): superset of v2; adds teacher_sign_ii and
         teacher_sign_tmag, plus an internal magnitude-source override.
  vanilla_regressor_v4.py  (v4, E11): superset of v3; adds opposite_mode
         force_ii_tmag and an ORTHOGONAL flag same_side_tmag.
opposite_mode=None AND same_side_tmag=False reproduces v1 bit-for-bit in every
version (verified each run: the "uniform" arm matches across E8/E9/E10).
''')


# ---------------------------------------------------------------------------
_s("5. opposite_mode catalogue (precise semantics)", '''
All of these change ONLY the opposite-side, teacher-fired samples (teacher fired
AND its error points the opposite way to the student's). Everything else --
teacher-didn't-fire samples and same-side samples -- keeps the normal sign-based
dispatch. So Type-I still does most of the work in training; these modes only
re-route that one "disagreement" slice.

  None / uniform     : blend by f_opposite (=0) -> teacher ignored on opposite side.
                       Magnitude & type from the student's error. This is the
                       baseline "teacher with opposite-side ignored".

  force_ii           : ALWAYS Type-II (lower), regardless of sign. Magnitude from
                       the student's error. Stable (Type-II self-limits).

  force_ii_tmag      : ALWAYS Type-II, but magnitude from the TEACHER's error
                       (pe_teacher). [E11 new]

  teacher_sign       : route by the TEACHER's sign --
                          pe_teacher > 0 -> Type-II ;  pe_teacher < 0 -> Type-I.
                       Magnitude from the student's error. UNSTABLE: can fire Type-I
                       (raise) on an already-too-high student -> runaway.

  teacher_sign_ii    : Type-II if pe_teacher > 0, else NO-OP ("skip"). Drops the
                       dangerous Type-I branch entirely. [E10 fix a]

  teacher_sign_tmag  : same routing as teacher_sign (incl. Type-I) BUT magnitude
                       from the teacher's frozen/bounded error -> the Type-I that
                       fires cannot self-compound. [E10 fix b; E10 winner]

A "skip" feedback state was added so a no-op opposite-side sample does NOT fall
back to the student's normal feedback.

Why magnitude-from-teacher matters: the teacher is FROZEN, so pe_teacher is a fixed
constant per sample across all epochs. Sourcing update_p from it bounds the step and
removes the self-amplifying term that drives runaways AND limits how much a BAD
teacher can hurt (it cannot scale its damage by the student's growing error).
''')


# ---------------------------------------------------------------------------
_s("6. same_side_tmag (orthogonal flag, E11)", '''
same_side_tmag=True : on SAME-side teacher-fired samples, take the update MAGNITUDE
from the teacher's error (pe_teacher) instead of the blended student error. The SIGN
is unchanged (student & teacher already agree on direction there), so only the step
size moves -- pulled to the teacher's bounded magnitude. Opposite-side untouched
(behaves like uniform). Isolates the question: does teacher-bounded magnitude help on
its own, with no type override at all?
''')


# ---------------------------------------------------------------------------
_s("7. Worked numerical examples", '''
Settings: T=5000, f=0.75, p_teacher=0.5. Assume the teacher's coin-flip FIRED.
Target encoded_Y = 3000 in every sample. update_p = (magnitude / 5000) ** 2.

SAMPLE A -- same-side (student & teacher both too high)
  pred_y=3200 -> error +200 ; teacher_pred=3500 -> pe_teacher +500.
  Same-side blend: error <- 0.25*200 + 0.75*500 = +425  -> still positive -> Type-II.
    uniform / force_ii / force_ii_tmag / teacher_sign_tmag :
        magnitude = blended 425 -> update_p = (425/5000)^2 = 0.00723 , Type-II
    sameside_tmag :
        magnitude = teacher 500 -> update_p = (500/5000)^2 = 0.01000 , Type-II
  => same_side_tmag changes ONLY the step (0.00723 -> 0.01); type/direction same.

SAMPLE B -- opposite, student too LOW, teacher too HIGH
  pred_y=2700 -> error -300 ; teacher_pred=3600 -> pe_teacher +600.
    uniform            : Type-I  (raise) , mag student 300 , update_p 0.0036  (teacher ignored)
    force_ii           : Type-II (lower) , mag student 300 , update_p 0.0036
    force_ii_tmag      : Type-II (lower) , mag teacher 600 , update_p 0.0144   (~4x stronger)
    teacher_sign_tmag  : Type-II (lower) , mag teacher 600 , update_p 0.0144   (pe>0 -> II)
    sameside_tmag      : Type-I  (raise) , mag student 300 , update_p 0.0036   (= uniform)

SAMPLE C -- opposite, student too HIGH, teacher too LOW  (THE RUNAWAY CASE)
  pred_y=3400 -> error +400 ; teacher_pred=2500 -> pe_teacher -500.
    teacher_sign (E9)      : Type-I (raise) on an already-too-high pred -> WRONG
                             direction; magnitude = student error, which GROWS each
                             epoch -> positive feedback loop -> RMSE blow-up (-70).
    teacher_sign_tmag (E10): Type-I (raise) still, but magnitude = teacher 500,
                             FROZEN -> capped step, cannot compound. Damage bounded.
    force_ii / force_ii_tmag: Type-II (lower) -> CORRECT direction here; safe.

This is exactly why teacher_sign exploded on airquality and the two E10 fixes
(drop Type-I, or bound the magnitude) removed it.
''')


# ---------------------------------------------------------------------------
_s("8. Experiments E8-E11 (purpose, config, findings)", '''
Common task: teacher 20% -> student 30% mask, structured, f=0.75, p=0.5, 100 epochs,
6 seeds, held-out TEST, paired by seed. Datasets: ccpp (teacher helps), airquality
(weak help), california (neutral/hurts), energy (bad teacher).

E8 -- adaptive teacher TRIGGER (experiment8_adaptive.py).
  Fire the teacher preferentially where teacher<->student DISAGREE (adapt_disagree)
  or where the student ERRS (adapt_error), budget-matched p_e = clip(p*d/mean(d),0,1).
  VERDICT: adaptive is a GAIN-MULTIPLIER / sharpener, not a fixer -- it amplifies the
  teacher's effect in BOTH directions. Best single result of the arc (ccpp adapt_error
  +0.496, t=4.64) but california/energy get significantly WORSE. adapt_error is the
  stronger amplifier than adapt_disagree.

E9 -- opposite-side feedback OVERRIDE (experiment9_opposite.py, v2).
  Modes uniform / force_ii / teacher_sign. Override the feedback TYPE on opposite-side.
  VERDICT: force_ii = safe + mildly useful -- turns weak-teacher airquality into a
  SIGNIFICANT win (+0.175, t=2.33), no blow-ups. teacher_sign = high ceiling (best on
  ccpp) but CATASTROPHICALLY unstable on airquality (mean -18, seeds to -80) via the
  wrong-direction Type-I runaway. Neither rescues a bad teacher (energy still hurts).

E10 -- teacher_sign STABILITY fixes (experiment10_opposite_fix.py, v3).
  Modes uniform / teacher_sign (ref) / teacher_sign_ii / teacher_sign_tmag.
  VERDICT: both fixes KILL the runaway (airquality std 25.4 -> ~0.23). fix (b)
  teacher_sign_tmag WINS: airquality +0.107 (positive, beats uniform), energy harm
  HALVED (-0.152 vs teacher_sign -0.377), lowest variance, ~uniform on ccpp/california.
  fix (a) teacher_sign_ii is merely safe (no blow-up, but no upside, still hurts
  energy). KEY INSIGHT: the transferable lever is teacher-BOUNDED MAGNITUDE -- the
  first mechanism in the arc to make a BAD teacher LESS harmful, not just amplify it.

E11 -- teacher-bounded magnitude GENERALISED (experiment11_tmag.py, v4). COMPLETE.
  Modes uniform / force_ii (ref) / force_ii_tmag (new) / teacher_sign_tmag (ref) /
  sameside_tmag (new). M00 (full-data ceiling) from full_baseline.py.
  VERDICT (outputs/distillation/exp11/tmag_verdict.md):
  (1) Bounded magnitude is a SHARPENER, not a universal lever -- PARTLY REFUTES E10.
      force_ii -> force_ii_tmag: ccpp +0.251 -> +0.431 (best of arc, 6/6 seeds) GOOD;
      airquality +0.175 -> -0.140 WIN DESTROYED; energy -0.237 -> -0.116 harm halved.
      i.e. amplifies a GOOD teacher, caps a BAD one, but BACKFIRES on a marginal teacher
      (airquality's win came from the STUDENT's magnitude). Same sharpener pattern as E8.
      Best good-teacher mode = force_ii_tmag (ccpp); gentlest bad-teacher mode =
      sameside_tmag (energy -0.013 ~ no-teacher). No mode is both.
  (2) vs FULL M00: see section 9(C).
''')


# ---------------------------------------------------------------------------
_s("9. Cross-cutting analyses (E9/E10 numbers)", '''
(A) VS NO-TEACHER (impr; + = beat training with no teacher). The teacher's value
    tracks the DATASET, not the mechanism:
      ccpp (good)      : ALL modes beat no-teacher (uniform +0.36 t2.6, teacher_sign
                         +0.43 t3.3, tmag +0.37 t2.7, ...). Use a teacher.
      airquality (weak): force_ii +0.175 (t2.3, best), tmag +0.107, uniform +0.078;
                         teacher_sign -18 (blows up).
      california (neut): NO mode beats no-teacher (all ~ -0.01..-0.02, sig but tiny).
      energy (bad)     : NO mode beats no-teacher (uniform least harm -0.062; overrides
                         worse; tmag -0.152 the gentlest override).
    => No mode is universally better than no-teacher; none rescue a bad teacher.

(B) VS THE TEACHER M20 (does the guided student SURPASS the teacher it learned from?
    Note the student has LESS data, 70% vs 80% kept -> a real win if it happens):
      ccpp      : all stable modes beat M20 (teacher_sign best 4.676 vs 4.879, 5/6
                  seeds; uniform/tmag/ii marginal 3/6; force_ii does NOT, 4.952).
      airquality: force_ii beats M20 (1.560 vs 1.657, 4/6); teacher_sign_tmag beats
                  (3/6); uniform ~ties.
      california/energy: NO mode beats M20 (energy gap large, 0/6).
    => Student-beats-teacher happens, but ONLY where guidance already helps. The only
       STABLE mode beating M20 on BOTH ccpp+airquality is teacher_sign_tmag.

(C) VS THE FULL-DATA MODEL M00 (E11). M00 RMSE: ccpp 4.758, airquality 1.745,
    california 0.612, energy 0.959.
      ccpp       : force_ii_tmag 4.772 TIES M00 (within noise, 3/6 seeds) -- a 70%-data
                   student + distillation nearly matches the full-data model. Doesn't beat.
      airquality : force_ii BEATS M00 by +0.185 (5/6) -- BUT M00 is anomalously bad here
                   (M00 1.745 > M20 1.657 > 30%-student 1.736: MORE DATA = WORSE). The
                   structured holes remove harmful points; airquality is a LESS-IS-MORE
                   dataset. Direct support for the spacing/density hypothesis, not a win
                   over an honest ceiling.
      california : NO (gap ~0.02-0.04, full wins).
      energy     : NO (large gap, M00 0.959 vs best guided ~1.24).

(D) TEACHER-EDGE CORRELATION (correlate_teacher_edge.py). teacher_edge = RMSE(unguided)
    - RMSE(M20) (>0 = teacher is the more accurate model). Across 4 datasets:
      edge: california +0.007, airquality +0.078, energy +0.210, ccpp +0.323.
      Spearman rho = +0.40 (all methods), Pearson +0.33..+0.67 -> weakly positive but
      NOT predictive. ENERGY is the off-trend point: 2nd-highest edge, WORST benefit.
    => TEACHER ACCURACY DOES NOT PREDICT DISTILLATION BENEFIT. Aggregate accuracy is
       global; distillation injects the PER-SAMPLE error direction+magnitude into TA
       feedback -- a smoother/more-accurate teacher can still give per-sample nudges
       that conflict with what the masked-region samples need. So do NOT gate on the
       teacher edge (it would green-light energy, the worst case). The likelier
       predictor is the project's original data SPACING/DENSITY hypothesis.
''')


# ---------------------------------------------------------------------------
_s("10. Key conclusions so far", '''
1. Whether distillation helps is a property of the DATASET / the teacher's per-sample
   signal quality, NOT of the correction mechanism and NOT of the teacher's accuracy.
   No mode rescues a bad teacher (energy); all help a good one (ccpp).
2. The transferable mechanism-level lever is TEACHER-BOUNDED MAGNITUDE: sourcing
   update_p from the frozen teacher's error stabilises the update and is the only
   thing seen to make a BAD teacher LESS harmful (not just amplify it).
3. Type-II is the safe direction (self-limiting); mis-targeted Type-I is the source
   of instability. force_ii sidesteps it; teacher_sign_tmag tames it.
4. Practical recipe: GATE the teacher per dataset (only switch it on where confirmed
   to help). There, force_ii / teacher_sign_tmag extract the most; teacher_sign_tmag
   is the safest all-rounder (beats M20 on both productive datasets, never blows up).
5. A guided, data-starved student CAN surpass the teacher it learned from (ccpp,
   airquality) -- genuine distillation value-add, not imitation.
''')


# ---------------------------------------------------------------------------
_s("11. Files & how to run", '''
Experiments (WSL, ~/regexp-venv/bin/python <script>.py 2>/dev/null | tee outputs/_X.log):
  experiment8_adaptive.py       -> outputs/distillation/exp8/   (+ adaptive_verdict.md)
  experiment9_opposite.py       -> outputs/distillation/exp9/   (+ opposite_verdict.md)
  experiment10_opposite_fix.py  -> outputs/distillation/exp10/  (+ oppfix_verdict.md)
  experiment11_tmag.py          -> outputs/distillation/exp11/  (verdict pending)
  full_baseline.py              -> outputs/distillation/exp11/full_baseline.csv  (M00 only,
                                   the cheap full-data ceiling; ~1/8 the cost of E11)

Regressors: tmu/tmu/models/regression/vanilla_regressor{,_v2,_v3,_v4}.py (section 4).

Analysis (Windows, no tmu):
  python compare_references.py     --exp-dir outputs/distillation/exp11
        -> per mode: mean RMSE & per-seed wins vs unguided / M20 / M00
           (auto-merges full_baseline.csv for the M00 column).
  python correlate_teacher_edge.py --exp-dir outputs/distillation/exp11
        -> teacher_edge_corr.{png,md,csv}: edge vs benefit, Pearson/Spearman, scatter.

Per-dataset CSV columns: dataset, seed, rmse_unguided, [rmse_M00], rmse_M20,
then rmse_<mode> and impr_<mode> for each mode. impr_<mode> = rmse_unguided - rmse_<mode>.
''')


# ===========================================================================
def main(argv):
    if not argv:
        print(__doc__.strip())
        print("\nSections:")
        for title, _ in SECTIONS:
            print(f"  {title}")
        print("\nUse:  python explanations_of_runs.py <n|all>")
        return
    arg = argv[0]
    chosen = SECTIONS if arg == "all" else [SECTIONS[int(arg)]]
    for title, body in chosen:
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(body)
        print()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
