# Copyright (c) 2023 Ole-Christoffer Granmo
from tmu.models.base import SingleClauseBankMixin, SingleWeightBankMixin, TMBaseModel
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ============================================================================
# V3 — opposite-side feedback override, superset of V2 (Experiment 10).
# Identical to vanilla_regressor.py EXCEPT fit() takes `opposite_mode`, which on
# OPPOSITE-side teacher samples overrides the feedback TYPE (and optionally the
# magnitude source). opposite_mode=None reproduces the v1 path bit-for-bit.
# Modes: None | "force_ii" | "teacher_sign" (V2) plus the two E10 stability fixes
#   "teacher_sign_ii"   and  "teacher_sign_tmag"  (see fit()).
# ============================================================================
from tmu.weight_bank import WeightBank
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)

# --- local-build sentinel -------------------------------------------------
print(f"[tmu LOCAL BUILD] vanilla_regressor_V3 loaded from: {__file__}")


class TMRegressor(TMBaseModel, SingleClauseBankMixin, SingleWeightBankMixin):

    max_y: float
    min_y: float

    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            platform='CPU',
            patch_dim=None,
            feature_negation=True,
            boost_true_positive_feedback=1,
            reuse_random_feedback=0,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            weighted_clauses=False,
            clause_drop_p=0.0,
            literal_drop_p=0.0,
            seed=None
    ):
        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            platform=platform,
            patch_dim=patch_dim,
            feature_negation=feature_negation,
            boost_true_positive_feedback=boost_true_positive_feedback,
            reuse_random_feedback=reuse_random_feedback,
            max_included_literals=max_included_literals,
            number_of_state_bits_ta=number_of_state_bits_ta,
            weighted_clauses=weighted_clauses,
            clause_drop_p=clause_drop_p,
            literal_drop_p=literal_drop_p,
            seed=seed
        )
        SingleClauseBankMixin.__init__(self)
        SingleWeightBankMixin.__init__(self)

    def init_clause_bank(self, X: np.ndarray, Y: np.ndarray):
        clause_bank_type, clause_bank_args = self.build_clause_bank(X=X)
        self.clause_bank = clause_bank_type(**clause_bank_args)

        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals

    def init_weight_bank(self, X: np.ndarray, Y: np.ndarray):
        self.weight_bank = WeightBank(
            np.ones(self.number_of_clauses).astype(np.int32)
        )

    def init_after(self, X: np.ndarray, Y: np.ndarray):
        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

    def fit(self, X, Y, shuffle=True,
            teacher_pred_encoded=None,
            f=0.0, f_opposite=0.0, p_teacher=0.0, teacher_rng=None,
            teacher_p_sample=None, opposite_mode=None,
            global_y_min=None, global_y_max=None,
            *args, **kwargs):
        # --- teacher-guided distillation params (all default to OFF, i.e. this
        #     reduces to the stock training path bit-for-bit). See
        #     experiment5_distillation.py.
        #   teacher_pred_encoded : precomputed frozen-teacher ENCODED predictions.
        #   f / f_opposite   : blend fraction toward the teacher error when student
        #                      and teacher errors are on the SAME / OPPOSITE side of
        #                      encoded_Y[e]  (f_opposite=1 == hard replace; unstable).
        #   opposite_mode    : how to handle OPPOSITE-side samples INSTEAD of the
        #                      f_opposite blend. On those samples the feedback TYPE is
        #                      overridden:
        #                        None             -> blend by f_opposite (v1 behaviour).
        #                        "force_ii"       -> always Type-II (V2).
        #                        "teacher_sign"   -> route by the TEACHER's error sign:
        #                                            Type-II if teacher over-predicts,
        #                                            Type-I if it under-predicts (V2;
        #                                            the Type-I branch can run away).
        #                        "teacher_sign_ii"-> (E10 fix a) Type-II when teacher
        #                                            over-predicts, NO-OP when it under-
        #                                            predicts. Drops the Type-I branch.
        #                        "teacher_sign_tmag"-> (E10 fix b) same routing as
        #                                            teacher_sign (incl. Type-I) but the
        #                                            update MAGNITUDE comes from the
        #                                            TEACHER's (frozen, bounded) error,
        #                                            killing the self-compounding runaway.
        #                      All keep the student's magnitude EXCEPT teacher_sign_tmag.
        #   p_teacher        : per-sample probability of applying the correction.
        #   teacher_rng      : dedicated RNG for the trigger coin-flip.
        #   global_y_min/max : shared y-encoding range across models.
        self.init(X, Y)

        if global_y_min is not None:
            self.min_y = global_y_min
            self.max_y = global_y_max

        if teacher_pred_encoded is not None and teacher_rng is None:
            teacher_rng = np.random.RandomState(0)

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(X)
            self.X_train = X.copy()

        if self.max_y - self.min_y == 0:
            encoded_Y = np.ascontiguousarray(np.zeros(Y.shape[0], dtype=np.int32))
        else:
            encoded_Y = np.ascontiguousarray(((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32))

        # Drops clauses randomly based on clause drop probability
        clause_active = (self.rng.rand(self.number_of_clauses) >= self.clause_drop_p).astype(np.int32)

        # Literals are dropped based on literal drop probability
        literal_active = np.zeros(self.clause_bank.number_of_ta_chunks, dtype=np.uint32)
        literal_active_integer = self.rng.rand(self.clause_bank.number_of_literals) >= self.literal_drop_p
        for k in range(self.clause_bank.number_of_literals):
            if literal_active_integer[k] == 1:
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] |= (1 << chunk_pos)

        if not self.feature_negation:
            for k in range(self.clause_bank.number_of_literals // 2, self.clause_bank.number_of_literals):
                ta_chunk = k // 32
                chunk_pos = k % 32
                literal_active[ta_chunk] &= (~(1 << chunk_pos))

        literal_active = literal_active.astype(np.uint32)

        shuffled_index = np.arange(X.shape[0])
        if shuffle:
            self.rng.shuffle(shuffled_index)

        for e in shuffled_index:
            clause_outputs = self.clause_bank.calculate_clause_outputs_update(literal_active, self.encoded_X_train, e)

            pred_y = np.dot(clause_active * self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            pred_y = np.clip(pred_y, 0, self.T)
            prediction_error = pred_y - encoded_Y[e]

            # --- teacher-guided correction (no-op unless a teacher is supplied
            #     AND p_teacher fires for this sample) ---
            _p_e = p_teacher if teacher_p_sample is None else teacher_p_sample[e]
            forced_feedback = None      # None -> dispatch by error sign; "i"/"ii"/"skip" -> forced
            mag_override = None          # None -> magnitude from prediction_error; else this value
            if teacher_pred_encoded is not None and _p_e > 0.0 and teacher_rng.rand() < _p_e:
                pe_teacher = teacher_pred_encoded[e] - encoded_Y[e]
                if (prediction_error >= 0) == (pe_teacher >= 0):   # same side of encoded_Y
                    prediction_error = (1.0 - f) * prediction_error + f * pe_teacher
                elif opposite_mode == "force_ii":                  # always Type II
                    forced_feedback = "ii"
                elif opposite_mode == "teacher_sign":              # route by teacher sign
                    forced_feedback = "ii" if pe_teacher > 0 else ("i" if pe_teacher < 0 else "skip")
                elif opposite_mode == "teacher_sign_ii":           # (E10 a) Type-II-only, else no-op
                    forced_feedback = "ii" if pe_teacher > 0 else "skip"
                elif opposite_mode == "teacher_sign_tmag":         # (E10 b) teacher sign + teacher magnitude
                    forced_feedback = "ii" if pe_teacher > 0 else ("i" if pe_teacher < 0 else "skip")
                    mag_override = pe_teacher
                else:                                              # blend / replace (v1 default)
                    prediction_error = (1.0 - f_opposite) * prediction_error + f_opposite * pe_teacher

            mag = prediction_error if mag_override is None else mag_override
            update_p = (1.0 * mag / self.T) ** 2

            # feedback type: forced on opposite-side teacher samples (opposite_mode),
            # otherwise follows the (possibly teacher-adjusted) error sign. "skip" -> neither.
            do_type_i = (forced_feedback == "i") or (forced_feedback is None and prediction_error < 0)
            do_type_ii = (forced_feedback == "ii") or (forced_feedback is None and prediction_error > 0)

            if do_type_i:
                self.clause_bank.type_i_feedback(
                    update_p=update_p,
                    clause_active=clause_active,
                    literal_active=literal_active,
                    encoded_X=self.encoded_X_train,
                    e=e
                )
                if self.weighted_clauses:
                    self.weight_bank.increment(
                        clause_outputs,
                        update_p,
                        clause_active,
                        False
                    )
            elif do_type_ii:
                self.clause_bank.type_ii_feedback(
                    update_p,
                    clause_active,
                    literal_active,
                    self.encoded_X_train,
                    e)

                if self.weighted_clauses:
                    self.weight_bank.decrement(
                        clause_outputs,
                        update_p,
                        clause_active,
                        False
                    )
        return

    def predict(self, X, **kwargs):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(X)
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0]))
        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            pred_y = np.dot(self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            Y[e] = 1.0 * pred_y * (self.max_y - self.min_y) / self.T + self.min_y
        return Y
