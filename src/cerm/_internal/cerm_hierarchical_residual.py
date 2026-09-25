from __future__ import annotations

"""Weighted-compatible facade for the historical hierarchical learner.

The complete historical implementation is preserved byte-for-byte in
``cerm_hierarchical_residual_legacy``. ``sample_weight=None`` and exact unit
weights execute that implementation directly. Only genuinely weighted fits use
the extensions defined here.
"""

from . import cerm_hierarchical_residual_legacy as _legacy
from .cerm_weighted_representation import (
    canonical_sample_weight,
    frequency_weighted_quantile,
)

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


class NestedQuantileEncoder(_legacy.NestedQuantileEncoder):
    """Historical nested encoder plus a frequency-weight fitting contract."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        X = np.asarray(X, dtype=float)
        weights = canonical_sample_weight(sample_weight, len(X))
        if weights is None:
            return super().fit(X, y)

        active = weights > 0.0
        fit_X = X[active]
        fit_weight = weights[active]
        self.feature_kinds_, self.feature_cardinalities_ = _normalize_semantics(
            X.shape[1], self.feature_kinds, self.feature_cardinalities
        )
        probabilities = np.arange(1, self.max_bins, dtype=float) / self.max_bins
        self.thresholds_ = []
        self.maps_ = {level: [] for level in self.levels}
        self.cardinalities_ = {
            level: np.zeros(X.shape[1], dtype=np.int16) for level in self.levels
        }
        self.direct_state_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.direct_state_cardinalities_ = np.zeros(X.shape[1], dtype=np.int32)
        map_cache: dict[tuple[str, int, int], np.ndarray] = {}

        for feature in range(X.shape[1]):
            kind = self.feature_kinds_[feature]
            is_direct = kind in _DIRECT_KINDS or kind in _ORDERED_STATE_KINDS
            if is_direct:
                values = fit_X[:, feature]
                rounded = np.rint(values).astype(np.int64)
                if not np.allclose(values, rounded):
                    raise ValueError(
                        f"finite-state feature {feature} contains non-integers"
                    )
                if np.any(rounded < 0):
                    raise ValueError("finite states must be non-negative")
                configured = self.feature_cardinalities_[feature]
                fine_card = int(configured or (rounded.max(initial=0) + 1))
                if rounded.max(initial=0) >= fine_card:
                    raise ValueError("observed state exceeds configured cardinality")
                self.thresholds_.append(np.empty(0, dtype=np.float64))
                self.direct_state_mask_[feature] = True
                self.direct_state_cardinalities_[feature] = fine_card
                for level in self.levels:
                    map_kind = (
                        "nominal" if kind in _DIRECT_KINDS else "contiguous"
                    )
                    cache_key = (map_kind, fine_card, int(level))
                    if cache_key not in map_cache:
                        map_cache[cache_key] = (
                            self._nominal_map(fine_card, level)
                            if kind in _DIRECT_KINDS
                            else self._contiguous_map(fine_card, level)
                        )
                    mapping = map_cache[cache_key]
                    self.maps_[level].append(mapping)
                    self.cardinalities_[level][feature] = (
                        int(mapping.max(initial=0)) + 1
                    )
                continue

            values = fit_X[:, feature]
            finite_mask = np.isfinite(values)
            finite = values[finite_mask]
            finite_weight = fit_weight[finite_mask]
            if finite.size == 0:
                thresholds = np.empty(0, dtype=np.float64)
            else:
                unique = _unique_if_at_most(finite, self.max_bins)
                if unique is not None and len(unique) <= self.max_bins:
                    thresholds = (unique[:-1] + unique[1:]) / 2.0
                else:
                    thresholds = np.unique(
                        frequency_weighted_quantile(
                            finite,
                            probabilities,
                            finite_weight,
                            is_prefiltered=True,
                        )
                    )
                    minimum = float(finite.min())
                    maximum = float(finite.max())
                    thresholds = thresholds[
                        (thresholds > minimum) & (thresholds < maximum)
                    ]
            thresholds = np.asarray(thresholds, dtype=np.float64)
            self.thresholds_.append(thresholds)
            fine_card = len(thresholds) + 1
            for level in self.levels:
                cache_key = ("contiguous", fine_card, int(level))
                if cache_key not in map_cache:
                    map_cache[cache_key] = self._contiguous_map(
                        fine_card, level
                    )
                mapping = map_cache[cache_key]
                self.maps_[level].append(mapping)
                self.cardinalities_[level][feature] = (
                    int(mapping.max(initial=0)) + 1
                )

        self.n_features_in_ = X.shape[1]
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        X = np.asarray(X, dtype=float)
        weights = canonical_sample_weight(sample_weight, len(X))
        if weights is None:
            return super().fit_transform(X, y)
        self.fit(X, y, sample_weight=weights)
        return self.transform(X)


class NewtonNestedEncoder(_legacy.NewtonNestedEncoder):
    """Historical target-dependent Newton encoder; weighting remains unchanged."""


class HierarchicalResidualCERM(_legacy.HierarchicalResidualCERM):
    """Historical binary learner with weighted quantile representation fitting."""

    def _new_encoder(self):
        kwargs = dict(
            max_bins=self.max_bins,
            levels=self.levels,
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
        )
        weighted = bool(
            getattr(self, "_weighted_representation_active_", False)
        )
        if self.encoder_kind == "newton":
            if not weighted:
                return _legacy.NewtonNestedEncoder(
                    prebins=self.newton_prebins,
                    gain_l2=self.newton_gain_l2,
                    min_hessian=self.newton_min_hessian,
                    **kwargs,
                )
            return NewtonNestedEncoder(
                prebins=self.newton_prebins,
                gain_l2=self.newton_gain_l2,
                min_hessian=self.newton_min_hessian,
                **kwargs,
            )
        if not weighted:
            return _legacy.NestedQuantileEncoder(**kwargs)
        return NestedQuantileEncoder(**kwargs)

    def _selection_sample(self, X, y, sample_weight=None, *, seed_offset=0):
        if sample_weight is None:
            return super()._selection_sample(
                X, y, None, seed_offset=seed_offset
            )
        weights = canonical_sample_weight(sample_weight, len(X))
        if weights is None:
            return super()._selection_sample(
                X, y, None, seed_offset=seed_offset
            )
        if np.any(weights == 0.0):
            active = weights > 0.0
            X = np.asarray(X)[active]
            y = np.asarray(y)[active]
            weights = weights[active]
            weights = canonical_sample_weight(weights, len(weights))
        return super()._selection_sample(
            X, y, weights, seed_offset=seed_offset
        )

    def _fit_structure(self, X, y, cfg, sample_weight=None):
        if sample_weight is None:
            self._weighted_representation_active_ = False
            return super()._fit_structure(X, y, cfg, None)
        weights = canonical_sample_weight(sample_weight, len(X))
        if weights is None:
            self._weighted_representation_active_ = False
            return super()._fit_structure(X, y, cfg, None)
        sample_weight = weights
        self._weighted_representation_active_ = True

        self.encoder_ = self._new_encoder()
        if X.shape[1] >= 512:
            if self.encoder_kind == "quantile":
                self.encoder_.fit(X, y, sample_weight=sample_weight)
            else:
                self.encoder_.fit(X, y)
            ranking_states = self.encoder_.transform_level_columns(
                X, self.ranking_level, range(X.shape[1])
            )
            self.feature_idx_ = _select_features(
                ranking_states, y, self.max_features, sample_weight
            )
            states = self.encoder_.transform_columns(X, self.feature_idx_)
        else:
            if self.encoder_kind == "quantile":
                states_all = self.encoder_.fit_transform(
                    X, y, sample_weight=sample_weight
                )
            else:
                states_all = self.encoder_.fit_transform(X, y)
            self.feature_idx_ = _select_features(
                states_all[self.ranking_level],
                y,
                self.max_features,
                sample_weight,
            )
            states = {
                level: values[:, self.feature_idx_]
                for level, values in states_all.items()
            }
        rank = self._rank(
            states[self.ranking_level],
            y,
            max(cfg.n_pairs, 1),
            sample_weight,
        )
        self.pairs_ = rank[: cfg.n_pairs]
        self.fine_pairs_ = self.pairs_[
            : min(cfg.n_fine_pairs, len(self.pairs_))
        ]
        self.config_ = cfg
        self._prepare_residual_maps()
        codes = self._build_codes_from_states(states)
        self.oh_ = ReferenceStateEncoder()
        design = self.oh_.fit_transform(codes)
        if getattr(self, "_retain_fit_training_graph", False):
            self._fit_training_states_ = states
            self._fit_training_codes_ = codes
            self._fit_training_design_ = design
        self.clf_ = fit_binary_logistic_exact(
            design,
            y,
            C=cfg.C,
            random_state=self.random_state,
            max_iter=1500,
            sample_weight=sample_weight,
        )
        self.classes_ = self.clf_.classes_
        self.design_dim_ = int(design.shape[1])
        self.nonzero_coef_ = int(
            np.count_nonzero(np.abs(self.clf_.coef_) > 1e-10)
        )
        self._compile_lookup()
        self._prepare_execution_maps()
        self.operator_columns_ = int(codes.shape[1])
        self.lookup_bytes_ = int(sum(table.nbytes for table in self.lookup_))
        self.model_bytes_estimate_ = int(
            self.lookup_bytes_
            + self.encoder_.threshold_bytes_
            + len(self.feature_idx_) * 2
            + (len(self.pairs_) + len(self.fine_pairs_)) * 4
        )
        return self

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self._weighted_representation_active_ = False
            return super().fit(X, y, None)

        X = np.asarray(X, float)
        y = np.asarray(y, int)
        sample_weight = canonical_sample_weight(sample_weight, len(X))
        if sample_weight is None:
            self._weighted_representation_active_ = False
            return super().fit(X, y, None)
        self._weighted_representation_active_ = True

        Xsel, ysel, wsel = self._selection_sample(X, y, sample_weight)
        ia, iv = train_test_split(
            np.arange(len(Xsel)),
            test_size=0.22,
            stratify=ysel,
            random_state=self.random_state,
        )
        Xa, Xv, ya, yv = Xsel[ia], Xsel[iv], ysel[ia], ysel[iv]
        wa = None if wsel is None else wsel[ia]
        wv = None if wsel is None else wsel[iv]

        if self.max_bins == 16:
            candidates = [
                HierConfig(0, 0, 8, 0.2),
                HierConfig(0, 0, 16, 0.2),
                HierConfig(0, 0, 16, 1.0),
                HierConfig(10, 0, 16, 0.2),
                HierConfig(10, 0, 16, 1.0),
                HierConfig(30, 0, 16, 0.2),
                HierConfig(30, 0, 16, 1.0),
                HierConfig(30, 5, 16, 0.2),
                HierConfig(30, 5, 16, 1.0),
            ]
        elif self.max_bins == 8:
            candidates = [
                HierConfig(0, 0, 4, 0.2),
                HierConfig(0, 0, 8, 0.2),
                HierConfig(0, 0, 8, 1.0),
                HierConfig(10, 0, 8, 0.2),
                HierConfig(10, 0, 8, 1.0),
                HierConfig(30, 0, 8, 0.2),
                HierConfig(30, 0, 8, 1.0),
            ]
        else:
            candidates = [
                HierConfig(0, 0, 4, 0.2),
                HierConfig(0, 0, 4, 1.0),
                HierConfig(10, 0, 4, 0.2),
                HierConfig(10, 0, 4, 1.0),
                HierConfig(30, 0, 4, 0.2),
                HierConfig(30, 0, 4, 1.0),
            ]
        if self.fixed_C is not None:
            candidates = list(
                dict.fromkeys(
                    HierConfig(
                        config.n_pairs,
                        config.n_fine_pairs,
                        config.max_main_level,
                        self.fixed_C,
                    )
                    for config in candidates
                )
            )

        encoder = self._new_encoder()
        if Xa.shape[1] >= 512:
            if self.encoder_kind == "quantile":
                encoder.fit(Xa, ya, sample_weight=wa)
            else:
                encoder.fit(Xa, ya)
            selection_states = encoder.transform_level_columns(
                Xa, self.ranking_level, range(Xa.shape[1])
            )
            feature_idx = _select_features(
                selection_states, ya, self.max_features, wa
            )
            states_train = encoder.transform_columns(Xa, feature_idx)
            states_valid = encoder.transform_columns(Xv, feature_idx)
        else:
            if self.encoder_kind == "quantile":
                states_all = encoder.fit_transform(
                    Xa, ya, sample_weight=wa
                )
            else:
                states_all = encoder.fit_transform(Xa, ya)
            valid_all = encoder.transform(Xv)
            feature_idx = _select_features(
                states_all[self.ranking_level], ya, self.max_features, wa
            )
            states_train = {
                level: values[:, feature_idx]
                for level, values in states_all.items()
            }
            states_valid = {
                level: values[:, feature_idx]
                for level, values in valid_all.items()
            }

        max_pairs = max(config.n_pairs for config in candidates)
        rank = self._rank(
            states_train[self.ranking_level],
            ya,
            max(max_pairs, 1),
            wa,
        )
        cache = object.__new__(HierarchicalResidualCERM)
        cache.__dict__.update(self.__dict__)
        cache.encoder_ = encoder
        cache.feature_idx_ = feature_idx
        max_pair_count = min(max_pairs, len(rank))
        max_fine = max(config.n_fine_pairs for config in candidates)
        max_config = HierConfig(
            max_pair_count,
            min(max_fine, max_pair_count),
            self.max_bins,
            0.2,
        )
        cache.config_ = max_config
        cache.pairs_ = rank[:max_pair_count]
        cache.fine_pairs_ = cache.pairs_[: max_config.n_fine_pairs]
        cache._prepare_residual_maps()
        max_train_codes = cache._build_codes_from_states(states_train)
        max_valid_codes = cache._build_codes_from_states(states_valid)
        bank = EncodedColumnBank.build(
            max_train_codes, max_valid_codes, ReferenceStateEncoder
        )
        metadata = self._maximal_code_metadata(
            len(feature_idx), len(cache.pairs_), len(cache.fine_pairs_)
        )
        if len(metadata) != max_train_codes.shape[1]:
            raise RuntimeError("maximal hierarchical code metadata mismatch")

        design_groups = {}
        for config in candidates:
            key = (
                config.n_pairs,
                config.n_fine_pairs,
                config.max_main_level,
            )
            design_groups.setdefault(key, []).append(config)
        groups = list(design_groups.values())
        group_workers = min(
            max(1, effective_n_jobs(self.n_jobs)), len(groups)
        )

        def evaluate_group(group):
            representative = group[0]
            code_columns = self._candidate_code_columns(metadata, representative)
            design_train, design_valid = bank.view(code_columns)
            path = solve_binary_logistic_path(
                design_train,
                ya,
                design_valid,
                [config.C for config in group],
                random_state=self.random_state,
                max_iter=1500,
                sample_weight=wa,
                n_jobs=1 if group_workers > 1 else self.n_jobs,
            )
            actual_pairs = min(representative.n_pairs, len(cache.pairs_))
            actual_fine = min(
                representative.n_fine_pairs,
                actual_pairs,
                len(cache.fine_pairs_),
            )
            lookup_bytes = bank.lookup_bytes(code_columns)
            bytes_estimate = int(
                lookup_bytes
                + encoder.threshold_bytes_
                + len(feature_idx) * 2
                + (actual_pairs + actual_fine) * 4
            )
            operators = int(len(code_columns))
            rows = []
            for config in group:
                probability = np.clip(
                    path[float(config.C)].valid_probability,
                    1e-10,
                    1 - 1e-10,
                )
                losses = -(
                    yv * np.log(probability)
                    + (1 - yv) * np.log(1 - probability)
                )
                mean_loss = float(
                    np.mean(losses)
                    if wv is None
                    else np.average(losses, weights=wv)
                )
                if wv is None:
                    se_loss = float(
                        losses.std(ddof=1) / math.sqrt(len(losses))
                    )
                else:
                    centered = losses - mean_loss
                    se_loss = float(
                        np.sqrt(
                            np.average(centered * centered, weights=wv)
                            / max(np.count_nonzero(wv), 1)
                        )
                    )
                objective = float(
                    mean_loss
                    + self.cost_per_byte * bytes_estimate
                    + self.cost_per_operator * operators
                )
                rows.append(
                    dict(
                        mean_loss=mean_loss,
                        se_loss=se_loss,
                        objective=objective,
                        cfg=config,
                        bytes=bytes_estimate,
                        operators=operators,
                        design_dim=int(design_train.shape[1]),
                    )
                )
            return rows

        if group_workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=group_workers) as pool:
                group_rows = list(pool.map(evaluate_group, groups))
            scores = [row for rows in group_rows for row in rows]
        else:
            scores = []
            for group in groups:
                scores.extend(evaluate_group(group))
        scores.sort(key=lambda row: (row["objective"], row["mean_loss"]))
        self.validation_scores_ = scores
        self.training_graph_columns_ = int(max_train_codes.shape[1])
        self.training_graph_design_dim_ = int(bank.train.shape[1])
        self.selection_rows_ = int(len(Xsel))
        if self.selection_rule == "best":
            chosen = scores[0]
        else:
            best = min(scores, key=lambda row: row["mean_loss"])
            tolerance = (
                best["se_loss"]
                if self.selection_rule == "one_se_cost"
                else min(0.0025, 0.25 * best["se_loss"])
            )
            eligible = [
                row
                for row in scores
                if row["mean_loss"] <= best["mean_loss"] + tolerance
            ]
            chosen = min(
                eligible,
                key=lambda row: (
                    row["bytes"],
                    row["operators"],
                    row["mean_loss"],
                ),
            )
        self.best_config_ = chosen["cfg"]
        self.selected_validation_loss_ = chosen["mean_loss"]
        self.selected_validation_bytes_ = chosen["bytes"]
        return self._fit_structure(X, y, self.best_config_, sample_weight)


HierarchicalResidualCERM.operator_metadata = (
    _legacy.HierarchicalResidualCERM.operator_metadata
)
