import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def _sanitize_params(params):
    clean = {}

    for k, v in params.items():
        # convert numpy scalars
        if isinstance(v, (np.generic,)):
            v = v.item()
        # convert strings that look numeric
        if isinstance(v, str):
            if v.isdigit():
                v = int(v)
            else:
                try:
                    f = float(v)
                    if f.is_integer():
                        v = int(f)
                    else:
                        v = f
                except ValueError:
                    pass
        # ensure Python scalar, not numpy
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        clean[k] = v
    return clean


class LightGBMRegressorCV:
    def __init__(
        self,
        X,
        y,
        params,
        n_splits=5,
        random_state=42,
        transform="none",  # "none" or "log1p"
        clip_pred_lower=None,
        # --- hurdle options ---
        hurdle=False,
        clf_params=None,  # LightGBM params for classifier part
        hurdle_threshold=0.0,  # define y>threshold as "positive"
    ):
        """
        Cross-validated LightGBM regressor with optional target transform and optional hurdle modeling.

        If hurdle=False: behaves like your original regressor (single-stage).
        If hurdle=True: fits LGBMClassifier on (y>threshold) and a conditional regressor on y|y>threshold.
                        Combined preds are P(y>thr)*E[y|y>thr].

        Args common with previous version omitted for brevity.
        """
        self.X = X.copy()
        self.y = y.copy()
        self.params = params
        self.n_splits = n_splits
        self.random_state = random_state
        self.transform = transform
        self.clip_pred_lower = clip_pred_lower

        # hurdle config
        self.hurdle = hurdle
        self.clf_params = clf_params or dict(
            objective="binary", learning_rate=0.05, num_leaves=31)
        self.hurdle_threshold = hurdle_threshold

        # Outputs
        # predictions on original scale (combined if hurdle=True)
        self.oof_pred = None
        # (single-stage only) transformed-scale preds
        self.oof_pred_transformed = None
        self.feature_importance = pd.DataFrame()  # regressor importances
        # classifier importances (when hurdle=True)
        self.feature_importance_clf = pd.DataFrame()

        # models
        # regression boosters (lgb.train) or LGBMRegressor (hurdle)
        self._fold_models = []
        self._fold_clf = []  # classifier models when hurdle=True

        # additional OOF when hurdle=True
        self.oof_p = None  # P(y>thr)
        self.oof_mu = None  # E[y|y>thr]

        # choose transform/inverse
        if self.transform == "none":
            self._forward = lambda a: a
            self._inverse = lambda a: a
        elif self.transform == "log1p":
            self._forward = np.log1p
            self._inverse = np.expm1
        else:
            raise ValueError("transform must be 'none' or 'log1p'")

    # =========================
    # FIT (CV)
    # =========================
    def fit(self):
        """Run K-Fold training and store OOF predictions and feature importance."""
        cat_cols = self.X.select_dtypes(
            include=["object", "category"]).columns.tolist()
        X_cat = self.X.copy()
        for c in X_cat.select_dtypes("object"):
            X_cat[c] = X_cat[c].astype("category")

        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)

        if not self.hurdle:
            # ---------- single-stage (original behavior) ----------
            oof_pred = np.zeros(len(X_cat), dtype=float)
            oof_pred_trans = np.zeros(len(X_cat), dtype=float)
            feature_importance = pd.DataFrame()
            y_all_trans = self._forward(self.y)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cat), 1):
                print(f"\nFold {fold}")
                X_tr, X_val = X_cat.iloc[tr_idx], X_cat.iloc[val_idx]
                y_tr_trans, y_val_trans = y_all_trans.iloc[tr_idx], y_all_trans.iloc[val_idx]

                dtr = lgb.Dataset(
                    X_tr, label=y_tr_trans, categorical_feature=cat_cols, free_raw_data=False
                )
                dval = lgb.Dataset(
                    X_val, label=y_val_trans, categorical_feature=cat_cols, free_raw_data=False
                )
                clean_params = _sanitize_params(self.params)
                booster = lgb.train(
                    clean_params,
                    dtr,
                    num_boost_round=1000,
                    valid_sets=[dval],
                    valid_names=["validation"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
                )
                self._fold_models.append(booster)

                val_pred_trans = booster.predict(
                    X_val, num_iteration=booster.best_iteration)
                val_pred = self._inverse(val_pred_trans)

                if self.clip_pred_lower is not None:
                    val_pred = np.clip(val_pred, self.clip_pred_lower, None)

                oof_pred[val_idx] = val_pred
                oof_pred_trans[val_idx] = val_pred_trans

                fold_importance = pd.DataFrame(
                    {
                        "feature": booster.feature_name(),
                        "importance": booster.feature_importance(importance_type="gain"),
                        "fold": fold,
                    }
                )
                feature_importance = pd.concat(
                    [feature_importance, fold_importance], axis=0)

            self.oof_pred = oof_pred
            self.oof_pred_transformed = oof_pred_trans if self.transform != "none" else None
            self.feature_importance = feature_importance.reset_index(drop=True)
            return self

        else:
            # ---------- hurdle two-stage ----------
            oof_p = np.zeros(len(X_cat), dtype=float)  # P(y>thr)
            oof_mu = np.zeros(len(X_cat), dtype=float)  # E[y|y>thr]
            oof_final = np.zeros(len(X_cat), dtype=float)

            feat_imp_reg = pd.DataFrame()
            feat_imp_clf = pd.DataFrame()

            y = self.y
            thr = self.hurdle_threshold

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cat), 1):
                print(f"\nFold {fold}")
                X_tr, X_val = X_cat.iloc[tr_idx], X_cat.iloc[val_idx]
                y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                # 1) Classifier: y_bin = 1 if y>thr else 0
                y_tr_bin = (y_tr > thr).astype(int)
                clf = LGBMClassifier(**self.clf_params)
                clf.fit(X_tr, y_tr_bin, categorical_feature=cat_cols)
                p_val = clf.predict_proba(X_val)[:, 1]
                self._fold_clf.append(clf)

                # collect clf feature importance (split-importance)
                fi_clf = pd.DataFrame(
                    {
                        "feature": X_tr.columns,
                        "importance": clf.feature_importances_.astype(float),
                        "fold": fold,
                    }
                )
                feat_imp_clf = pd.concat([feat_imp_clf, fi_clf], axis=0)

                # 2) Conditional regressor on positives only (with transform)
                mask_pos = y_tr > thr
                X_tr_pos = X_tr.loc[mask_pos]
                y_tr_pos = self._forward(y_tr.loc[mask_pos])

                # use sklearn API for parity (can also keep lgb.train if you prefer)
                reg = LGBMRegressor(
                    **{**self.params, "objective": "regression"})
                reg.fit(X_tr_pos, y_tr_pos, categorical_feature=cat_cols)

                mu_val_trans = reg.predict(X_val)
                mu_val = self._inverse(mu_val_trans)
                if self.clip_pred_lower is not None:
                    mu_val = np.clip(mu_val, self.clip_pred_lower, None)

                oof_p[val_idx] = p_val
                oof_mu[val_idx] = mu_val
                oof_final[val_idx] = p_val * mu_val

                self._fold_models.append(reg)

                # collect reg feature importance
                fi_reg = pd.DataFrame(
                    {
                        "feature": X_tr_pos.columns,
                        "importance": reg.feature_importances_.astype(float),
                        "fold": fold,
                    }
                )
                feat_imp_reg = pd.concat([feat_imp_reg, fi_reg], axis=0)

            self.oof_p = oof_p
            self.oof_mu = oof_mu
            self.oof_pred = oof_final
            self.feature_importance = feat_imp_reg.reset_index(drop=True)
            self.feature_importance_clf = feat_imp_clf.reset_index(drop=True)
            return self

    # =========================
    # FIT SINGLE (user-provided splits)
    # =========================
    def fit_single(self, X_tr, y_tr, X_te, return_model=False):
        """
        Train once on the given (X_tr, y_tr) and predict on X_te.
        Returns preds on the original scale. Optionally returns the trained model(s).

        If hurdle=True: returns combined preds; if return_model=True returns (preds, (clf, reg))
        """
        X_tr = X_tr.copy()
        X_te = X_te.copy()
        for c in X_tr.select_dtypes("object"):
            X_tr[c] = X_tr[c].astype("category")
        for c in X_te.select_dtypes("object"):
            X_te[c] = X_te[c].astype("category")

        if not self.hurdle:
            # single-stage
            cat_cols = X_tr.select_dtypes(
                include=["category"]).columns.tolist()
            y_tr = pd.Series(y_tr)
            y_tr_trans = self._forward(y_tr)

            dtr = lgb.Dataset(
                X_tr, label=y_tr_trans, categorical_feature=cat_cols, free_raw_data=False
            )
            clean_params = _sanitize_params(self.params)

            booster = lgb.train(
                clean_params,
                dtr,
                num_boost_round=1000,
                valid_sets=[dtr],
                valid_names=["train"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
            )
            preds_trans = booster.predict(
                X_te, num_iteration=booster.best_iteration)
            preds = self._inverse(preds_trans)
            if self.clip_pred_lower is not None:
                preds = np.clip(preds, self.clip_pred_lower, None)
            return (preds, booster) if return_model else preds

        else:
            # hurdle two-stage
            thr = self.hurdle_threshold
            y_tr = pd.Series(y_tr)

            # 1) classifier
            y_tr_bin = (y_tr > thr).astype(int)
            clf = LGBMClassifier(**self.clf_params)
            clf.fit(
                X_tr,
                y_tr_bin,
                categorical_feature=X_tr.select_dtypes(
                    include=["category"]).columns.tolist(),
            )
            p_te = clf.predict_proba(X_te)[:, 1]

            # 2) conditional regressor
            mask_pos = y_tr > thr
            reg = LGBMRegressor(**{**self.params, "objective": "regression"})
            reg.fit(
                X_tr.loc[mask_pos],
                self._forward(y_tr.loc[mask_pos]),
                categorical_feature=X_tr.select_dtypes(
                    include=["category"]).columns.tolist(),
            )
            mu_te = self._inverse(reg.predict(X_te))
            if self.clip_pred_lower is not None:
                mu_te = np.clip(mu_te, self.clip_pred_lower, None)

            preds = p_te * mu_te
            return (preds, (clf, reg)) if return_model else preds

    def fit_single_by_idx(self, train_idx, test_idx, return_model=False):
        X_tr = self.X.iloc[train_idx]
        y_tr = self.y.iloc[train_idx]
        X_te = self.X.iloc[test_idx]
        return self.fit_single(X_tr, y_tr, X_te, return_model=return_model)

    # =========================
    # EVALUATE
    # =========================
    def evaluate(self, verbose=True):
        """
        If hurdle=False: metrics on self.oof_pred (single-stage).
        If hurdle=True: metrics on combined OOF (self.oof_pred = P * MU).
        """
        if self.hurdle:
            if self.oof_pred is None:
                raise ValueError("Model not trained yet. Call `.fit()` first.")
            rmse = np.sqrt(mean_squared_error(self.y, self.oof_pred))
            mae = mean_absolute_error(self.y, self.oof_pred)
            r2 = r2_score(self.y, self.oof_pred)
            if verbose:
                print(
                    f"OOF (hurdle combined) → RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
            return {"rmse": rmse, "mae": mae, "r2": r2}
        else:
            if self.oof_pred is None:
                raise ValueError("Model not trained yet. Call `.fit()` first.")
            rmse = np.sqrt(mean_squared_error(self.y, self.oof_pred))
            mae = mean_absolute_error(self.y, self.oof_pred)
            r2 = r2_score(self.y, self.oof_pred)
            results = {"rmse": rmse, "mae": mae, "r2": r2}
            if self.transform != "none":
                y_trans = self._forward(self.y)
                rmse_t = np.sqrt(mean_squared_error(
                    y_trans, self.oof_pred_transformed))
                mae_t = mean_absolute_error(y_trans, self.oof_pred_transformed)
                r2_t = r2_score(y_trans, self.oof_pred_transformed)
                results.update(
                    {"rmse_transformed": rmse_t,
                        "mae_transformed": mae_t, "r2_transformed": r2_t}
                )
                if verbose:
                    print(
                        f"OOF (original) → RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
                    print(
                        f"OOF (transformed '{self.transform}') → RMSE={rmse_t:.3f}, MAE={mae_t:.3f}, R²={r2_t:.3f}"
                    )
            else:
                if verbose:
                    print(
                        f"OOF (original) → RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
            return results

    # =========================
    # PREDICT
    # =========================
    def predict(self, X_new):
        """
        Predict on new data.
        - hurdle=False: returns single-stage predictions (original scale).
        - hurdle=True: returns combined predictions (P * MU).
        """
        Xp = X_new.copy()
        for c in Xp.select_dtypes("object"):
            Xp[c] = Xp[c].astype("category")

        if not self.hurdle:
            if not self._fold_models:
                raise ValueError("Model not trained yet. Call `.fit()` first.")
            preds_trans = []
            for booster in self._fold_models:
                # booster is an lgb.Booster in single-stage CV
                preds_trans.append(booster.predict(
                    Xp, num_iteration=booster.best_iteration))
            mean_trans = np.mean(preds_trans, axis=0)
            preds = self._inverse(mean_trans)
            if self.clip_pred_lower is not None:
                preds = np.clip(preds, self.clip_pred_lower, None)
            return preds
        else:
            if not self._fold_models or not self._fold_clf:
                raise ValueError("Model not trained yet. Call `.fit()` first.")

            p_preds = []
            mu_preds = []
            for clf, reg in zip(self._fold_clf, self._fold_models):
                p_preds.append(clf.predict_proba(Xp)[:, 1])
                mu_preds.append(self._inverse(reg.predict(Xp)))

            p_mean = np.mean(p_preds, axis=0)
            mu_mean = np.mean(mu_preds, axis=0)
            if self.clip_pred_lower is not None:
                mu_mean = np.clip(mu_mean, self.clip_pred_lower, None)
            return p_mean * mu_mean

    def predict_components(self, X_new):
        """
        For hurdle=True, return (p, mu, combined). For hurdle=False, returns (None, pred, pred).
        """
        if not self.hurdle:
            yhat = self.predict(X_new)
            return None, yhat, yhat

        Xp = X_new.copy()
        for c in Xp.select_dtypes("object"):
            Xp[c] = Xp[c].astype("category")

        p_preds, mu_preds = [], []
        for clf, reg in zip(self._fold_clf, self._fold_models):
            p_preds.append(clf.predict_proba(Xp)[:, 1])
            mu_preds.append(self._inverse(reg.predict(Xp)))
        p_mean = np.mean(p_preds, axis=0)
        mu_mean = np.mean(mu_preds, axis=0)
        if self.clip_pred_lower is not None:
            mu_mean = np.clip(mu_mean, self.clip_pred_lower, None)
        return p_mean, mu_mean, p_mean * mu_mean

    # =========================
    # PLOTS (unchanged semantics; use combined OOF when hurdle=True)
    # =========================
    def plot_predictions(self):
        if self.oof_pred is None:
            raise ValueError("Run `.fit()` first.")
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y, self.oof_pred, alpha=0.3)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        title = (
            "Predicted vs Actual (OOF - Hurdle Combined)"
            if self.hurdle
            else "Predicted vs Actual (OOF)"
        )
        plt.title(title)
        a_min, a_max = float(self.y.min()), float(self.y.max())
        plt.plot([a_min, a_max], [a_min, a_max], color="red", linestyle="--")
        plt.show()

    def plot_residuals(self):
        if self.oof_pred is None:
            raise ValueError("Run `.fit()` first.")
        residuals = self.y - self.oof_pred
        plt.figure(figsize=(7, 5))
        plt.scatter(self.oof_pred, residuals, alpha=0.3)
        plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        title = (
            "Residuals vs Predicted (OOF - Hurdle Combined)"
            if self.hurdle
            else "Residuals vs Predicted (OOF)"
        )
        plt.title(title)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.title("Distribution of Residuals")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.show()

    def plot_importance(self, top_n=20):
        """
        Plot average feature importance for the *regressor*.
        If hurdle=True and you also want classifier importances, access `feature_importance_clf`.
        """
        if self.feature_importance.empty:
            raise ValueError(
                "No feature importance found. Run `.fit()` first.")
        importance_summary = (
            self.feature_importance.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        plt.figure(figsize=(8, 6))
        importance_summary.head(top_n).plot(kind="barh")
        plt.gca().invert_yaxis()
        title = f"Top {top_n} Feature Importance (Regressor)"
        plt.title(title)
        plt.xlabel("Importance")
        plt.show()
        return importance_summary


class STRRevenuePipeline:
    """
    3-stage STR pipeline with optional per-stage distillation:
      1) Price  (all listings)
      2) Occupancy (active-only if active_only=True)
      3) Revenue   (active-only; uses cross-features)
    Distillation lets you use review/host features at TRAIN time
    but not require them at inference.
    """

    def __init__(
        self,
        feature_cols,
        price_params,
        occ_params,
        rev_params,
        transforms=None,
        active_only=True,
        make_cross_features=True,
        # NEW: per-stage distillation config
        distill=None,
    ):
        self.feature_cols = feature_cols
        self.price_params = price_params
        self.occ_params = occ_params
        self.rev_params = rev_params
        self.transforms = transforms or {
            "price": "log1p", "occ": "log1p", "rev": "log1p"}
        self.active_only = active_only
        self.make_cross_features = make_cross_features

        self.distill = distill or {
            "price": {"enabled": False, "extra_cols": [], "blend_weight": 0.0},
            "occ": {"enabled": False, "extra_cols": [], "blend_weight": 0.0},
            "rev": {"enabled": False, "extra_cols": [], "blend_weight": 0.0},
        }

        # Models
        self.m_price = None
        self.m_occ = None
        self.m_rev = None

        # (optional) keep teachers + OOF for analysis
        self.teacher_price = None
        self.teacher_occ = None
        self.teacher_rev = None
        self._oof_teacher = {}

    # -------------------------------------------------------
    def _make_regressor(self, X, y, params, transform):
        """Utility to build LightGBMRegressorCV with standard options."""
        return LightGBMRegressorCV(
            X, y, params=params, transform=transform, hurdle=False, clip_pred_lower=0
        )

    def _maybe_distill(self, stage, df, target_col, base_params, transform):
        """
        If distillation enabled for `stage`, train a teacher on (feature_cols + extra_cols),
        compute OOF preds, and return (X_student, y_blend, teacher_model or None).
        Otherwise return (X_student, y_true, None).
        """
        cfg = self.distill.get(stage, {"enabled": False})
        X_student = df[self.feature_cols].copy()
        y_true = df[target_col].copy()

        if not cfg.get("enabled", False):
            return X_student, y_true, None

        extra_cols = cfg.get("extra_cols", [])
        blend_w = float(cfg.get("blend_weight", 0.5))
        # Train TEACHER
        use_cols = self.feature_cols + \
            [c for c in extra_cols if c in df.columns]
        X_teacher = df[use_cols].copy()

        print(
            f"\n[Distill:{stage}] Fitting TEACHER on {len(use_cols)} features "
            f"(blend_w={blend_w:.2f})"
        )
        teacher = self._make_regressor(
            X_teacher, y_true, base_params, transform)
        teacher.fit()
        teacher.evaluate(verbose=True)

        oof_teacher = teacher.oof_pred.copy()  # original scale
        self._oof_teacher[stage] = oof_teacher
        # blended target on original scale
        y_blend = (1.0 - blend_w) * y_true + blend_w * oof_teacher

        return X_student, y_blend, teacher

    # -------------------------------------------------------
    def fit(self, df_train, city_mask=None):
        """Fit price, occupancy, and revenue models on training data (with optional distillation)."""
        if city_mask is not None:
            df_train = df_train.loc[~city_mask].copy()
            print(
                f"[fit] Using training data excluding held-out slice → {df_train.shape}")

        # ---------- PRICE (all listings)
        Xp, yp, self.teacher_price = self._maybe_distill(
            "price", df_train, "price_capped", self.price_params, self.transforms["price"]
        )
        self.m_price = self._make_regressor(
            Xp, yp, self.price_params, self.transforms["price"])
        print("\n[Fitting PRICE student]")
        self.m_price.fit()
        self.m_price.evaluate(verbose=True)

        # ---------- OCCUPANCY (active-only optional)
        df_occ = (
            df_train[df_train["estimated_occupancy_l365d"] > 0].copy()
            if self.active_only
            else df_train.copy()
        )
        if self.active_only:
            print(
                f"\nRestricting OCC training to {len(df_occ)} active listings "
                f"({len(df_occ)/len(df_train)*100:.1f}% active)"
            )

        Xo, yo, self.teacher_occ = self._maybe_distill(
            "occ", df_occ, "estimated_occupancy_l365d", self.occ_params, self.transforms[
                "occ"]
        )
        self.m_occ = self._make_regressor(
            Xo, yo, self.occ_params, self.transforms["occ"])
        print("\n[Fitting OCCUPANCY student]")
        self.m_occ.fit()
        self.m_occ.evaluate(verbose=True)

        # ---------- REVENUE (active-only optional)
        df_rev = (
            df_train[df_train["estimated_occupancy_l365d"] > 0].copy()
            if self.active_only
            else df_train.copy()
        )

        # Base student features
        Xr_base = df_rev[self.feature_cols].copy()

        # Add cross features (built from student preds on ORIGINAL scale)
        if self.make_cross_features:
            price_pred_tr = self.m_price.predict(Xr_base)
            occ_pred_tr = self.m_occ.predict(Xr_base)
            Xr_base["occ_x_price"] = occ_pred_tr * price_pred_tr
            Xr_base["log_occ_x_price"] = np.log1p(Xr_base["occ_x_price"])

        # Distill target if enabled
        if self.distill.get("rev", {}).get("enabled", False):
            cfg = self.distill["rev"]
            extra_cols = [c for c in cfg.get(
                "extra_cols", []) if c in df_rev.columns]
            yr_true = df_rev["estimated_revenue_l365d"].copy()

            print(
                f"\n[Distill:rev] Fitting TEACHER on base+{len(extra_cols)} extra cols "
                f"(blend_w={cfg.get('blend_weight', 0.5):.2f})"
            )

            # Build teacher input = base student features (already has cross-features) + extra_cols
            X_teacher_rev = Xr_base.copy()
            for c in extra_cols:
                X_teacher_rev[c] = df_rev[c].values

            teacher_rev = self._make_regressor(
                X_teacher_rev, yr_true, self.rev_params, self.transforms["rev"]
            )
            teacher_rev.fit()
            teacher_rev.evaluate(verbose=True)

            self.teacher_rev = teacher_rev
            oof_rev_teacher = teacher_rev.oof_pred.copy()
            w = float(cfg.get("blend_weight", 0.5))
            yr_student = (1 - w) * yr_true + w * oof_rev_teacher
        else:
            yr_student = df_rev["estimated_revenue_l365d"].copy()

        Xr_student = Xr_base

        # Fit student revenue
        self.m_rev = self._make_regressor(
            Xr_student, yr_student, self.rev_params, self.transforms["rev"]
        )
        print("\n[Fitting REVENUE student]")
        self.m_rev.fit()
        print("[Train] Revenue OOF evaluation:")
        self.m_rev.evaluate(verbose=True)

    # -------------------------------------------------------
    def evaluate_holdout(self, df_holdout, label="holdout"):
        """Evaluate revenue model on a held-out slice (active-only if configured)."""
        if self.m_rev is None:
            raise ValueError("Models not trained yet.")

        df_eval = df_holdout.copy()
        if self.active_only:
            df_eval = df_eval[df_eval["estimated_occupancy_l365d"] > 0]
            print(
                f"\n[{label}] active listings: {len(df_eval)} ({len(df_eval)/len(df_holdout)*100:.1f}% active)"
            )

        X_eval = df_eval[self.feature_cols].copy()

        # student price/occ preds → cross features (original scale)
        price_pred = self.m_price.predict(X_eval)
        occ_pred = self.m_occ.predict(X_eval)
        if self.make_cross_features:
            X_eval["occ_x_price"] = occ_pred * price_pred
            X_eval["log_occ_x_price"] = np.log1p(X_eval["occ_x_price"])

        rev_pred = self.m_rev.predict(X_eval)

        y_true = df_eval["estimated_revenue_l365d"].values
        y_true_log = np.log1p(y_true)
        y_pred_log = np.log1p(rev_pred)

        r2 = r2_score(y_true_log, y_pred_log)
        mae = mean_absolute_error(y_true_log, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        corr = np.corrcoef(y_true_log, y_pred_log)[0, 1]

        print(f"\n[{label.upper()} RESULTS]")
        print(
            f"R²(log)={r2:.3f}  MAE_log={mae:.3f}  RMSE_log={rmse:.3f}  Corr={corr:.3f}")
        print(
            f"Revenue preds: min={rev_pred.min():.2f}, p50={np.median(rev_pred):.2f}, max={rev_pred.max():.2f}"
        )
        print(
            f"True revenue:  min={y_true.min():.2f}, p50={np.median(y_true):.2f}, max={y_true.max():.2f}"
        )

        return {"r2": r2, "mae_log": mae, "rmse_log": rmse, "corr": corr, "n_eval": len(df_eval)}

    def predict_all(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price, occupancy, and revenue predictions using trained submodels.

        Parameters
        ----------
        df_input : pd.DataFrame
            Input features (must include self.feature_cols)

        Returns
        -------
        pd.DataFrame
            Columns: ['price_pred', 'occ_pred', 'rev_pred']
            Returned on the *original scale* (inverse of log1p transforms)
        """
        X = df_input[self.feature_cols].copy()

        # --- Step 1: predict price and occupancy ---
        price_pred_log = self.m_price.predict(X)
        occ_pred_log = self.m_occ.predict(X)

        # Invert log1p transform if used
        if self.transforms.get("price") == "log1p":
            price_pred = np.expm1(price_pred_log)
        else:
            price_pred = price_pred_log

        if self.transforms.get("occ") == "log1p":
            occ_pred = np.expm1(occ_pred_log)
        else:
            occ_pred = occ_pred_log

        # --- Step 2: optional cross features ---
        if getattr(self, "make_cross_features", False):
            X["occ_x_price"] = occ_pred * price_pred
            X["log_occ_x_price"] = np.log1p(X["occ_x_price"])

        # --- Step 3: predict revenue ---
        rev_pred_log = self.m_rev.predict(X)
        rev_pred = np.expm1(rev_pred_log) if self.transforms.get(
            "rev") == "log1p" else rev_pred_log

        # --- Step 4: assemble DataFrame ---
        preds = pd.DataFrame(
            {"price_pred": price_pred, "occ_pred": occ_pred, "rev_pred": rev_pred},
            index=df_input.index,
        )

        return preds
