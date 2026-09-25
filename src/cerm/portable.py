from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .validation import reset_dataframe_index_if_needed, validate_dataframe_schema
from ._internal.cerm_typed_quotient_adapters_v4 import TypedAdapterOutput, _numeric_series_values


def _decode_scalar(payload: dict[str, Any]) -> Any:
    kind = payload.get("type")
    if kind in {"NoneType", "none"}:
        return None
    if kind in {"str", "bool", "int", "float"}:
        return {"str": str, "bool": bool, "int": int, "float": float}[kind](
            payload.get("value")
        )
    if kind == "bytes-hex":
        return bytes.fromhex(payload["value"])
    if kind == "tuple":
        return tuple(_decode_scalar(item) for item in payload.get("items", []))
    if kind == "timestamp":
        return pd.Timestamp(payload["value"])
    raise ValueError(
        "typed adapter contains an unsupported categorical value: "
        f"{payload.get('python_type', kind)!r}"
    )


def _mapping_items(
    payload: Any,
    *,
    value_plural: str,
    value_singular: str,
) -> list[tuple[Any, Any]]:
    """Decode both legacy row mappings and compact parallel mappings."""
    if isinstance(payload, dict):
        if payload.get("encoding") != "parallel-json-v1":
            raise ValueError(f"unsupported categorical mapping encoding: {payload.get('encoding')!r}")
        key_type = payload.get("key_type")
        keys = payload.get("keys", [])
        values = payload.get(value_plural, [])
        if len(keys) != len(values):
            raise ValueError("typed adapter categorical mapping arrays have different lengths")
        return [
            (_decode_scalar({"type": key_type, "value": key}), value)
            for key, value in zip(keys, values)
        ]
    return [
        (_decode_scalar(item["key"]), item[value_singular])
        for item in (payload or [])
    ]


@dataclass
class PortableTypedAdapter:
    """Inference-only typed adapter reconstructed from checked JSON/NPZ IR.

    Version 1 supports numeric imputation and missing indicators plus identity,
    ordered, and Newton categorical encoders. Embedding adapters deliberately
    remain non-portable until their complete runtime contract is versioned.
    """

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, json_path: str | Path, npz_path: str | Path) -> "PortableTypedAdapter":
        metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if metadata.get("format") != "cerm-typed-adapter-ir-v1":
            raise ValueError(f"unsupported typed adapter format: {metadata.get('format')!r}")
        if not metadata.get("portable", False):
            raise ValueError("typed adapter IR is not portable")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return cls(metadata=metadata, arrays=arrays)

    @property
    def input_columns(self) -> tuple[Any, ...]:
        return tuple(self.metadata["input_columns"])

    @property
    def nbytes(self) -> int:
        meta = len(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))
        return int(meta + sum(value.nbytes for value in self.arrays.values()))

    def export(self, prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        np.savez_compressed(npz_path, **self.arrays)
        metadata = dict(self.metadata)
        metadata["array_file"] = npz_path.name
        json_path.write_text(
            json.dumps(metadata, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        return npz_path, json_path

    def transform_columns(
        self, X: pd.DataFrame, output_indices
    ) -> TypedAdapterOutput:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, self.input_columns))
        indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        output_dim = len(self.metadata.get("output_feature_names", []))
        if np.any((indices < 0) | (indices >= output_dim)):
            raise IndexError("portable typed adapter output index out of range")
        if len(indices) == output_dim and np.array_equal(
            indices, np.arange(output_dim, dtype=np.int64)
        ):
            return self.transform(frame)

        wanted = {int(index): [] for index in indices}
        out_index = 0
        numeric_cache = {}
        category_cache = {}
        generated = {}

        for entry in self.metadata.get("numeric", []):
            column = self.input_columns[int(entry["column_index"])]
            slots = [(out_index, "numeric")]
            out_index += 1
            if entry.get("emit_missing_state", False):
                slots.append((out_index, "missing"))
                out_index += 1
            if not any(slot in wanted for slot, _ in slots):
                continue
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            median = float(self.arrays[entry["median_array"]][0])
            numeric_cache[column] = (np.where(missing, median, values), missing)
            for slot, kind in slots:
                if slot in wanted:
                    generated[slot] = (
                        numeric_cache[column][0]
                        if kind == "numeric"
                        else numeric_cache[column][1].astype(np.int16)
                    )

        for entry in self.metadata.get("categorical", []):
            column = self.input_columns[int(entry["column_index"])]
            mode = entry["mode"]
            if mode == "identity":
                representation = entry.get(
                    "representation", self.metadata.get("category_identity", "state")
                )
                width = (
                    max(1, int(np.ceil(np.log2(max(int(entry["cardinality"]), 2)))))
                    if representation == "binary" else 1
                )
            else:
                representation = "quotient"
                width = 1
            slots = list(range(out_index, out_index + width))
            out_index += width
            selected_slots = [slot for slot in slots if slot in wanted]
            if not selected_slots:
                continue
            series = frame[column]
            if mode == "identity":
                mapping = {
                    key: int(value)
                    for key, value in _mapping_items(
                        entry.get("mapping", []),
                        value_plural="states",
                        value_singular="state",
                    )
                }
                mapped = series.astype("object").map(mapping)
                fallback = int(entry.get("unknown_state", entry.get("missing_state", 0)))
                states = mapped.fillna(fallback).to_numpy(dtype=np.int16)
                if representation == "binary":
                    for bit, slot in enumerate(slots):
                        if slot in wanted:
                            generated[slot] = (states >> bit) & 1
                else:
                    generated[slots[0]] = states
            else:
                mapping = {
                    key: float(value)
                    for key, value in _mapping_items(
                        entry.get("mapping", []),
                        value_plural="scores",
                        value_singular="score",
                    )
                }
                scores = (
                    series.astype("object")
                    .map(mapping)
                    .fillna(float(entry["default_score"]))
                    .to_numpy(dtype=np.float64)
                )
                thresholds = self.arrays[entry["thresholds_array"]]
                states = np.searchsorted(thresholds, scores, side="right") + 1
                states[series.isna().to_numpy()] = 0
                generated[slots[0]] = states.astype(np.int16)

        if self.metadata.get("embedding") is not None:
            raise ValueError("portable typed adapter does not support embedding IR yet")
        matrix = (
            np.column_stack([generated[int(index)] for index in indices]).astype(
                np.float64, copy=False
            )
            if len(indices)
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        names = self.metadata.get("output_feature_names", [])
        kinds = self.metadata.get("output_feature_kinds", [])
        cards = self.metadata.get("output_cardinalities", [])
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=[names[int(i)] for i in indices],
            feature_kinds=[kinds[int(i)] for i in indices],
            cardinalities=[cards[int(i)] for i in indices],
            metadata={
                "adapter_version": self.metadata.get("adapter_version"),
                "portable": True,
                "projected": True,
            },
        ).validate(len(frame))

    def transform(self, X: pd.DataFrame) -> TypedAdapterOutput:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, self.input_columns))
        output: list[np.ndarray] = []

        for entry in self.metadata.get("numeric", []):
            column = self.input_columns[int(entry["column_index"])]
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            median = float(self.arrays[entry["median_array"]][0])
            output.append(np.where(missing, median, values)[:, None])
            if entry.get("emit_missing_state", False):
                output.append(missing.astype(np.int16)[:, None])

        for entry in self.metadata.get("categorical", []):
            column = self.input_columns[int(entry["column_index"])]
            series = frame[column]
            mode = entry["mode"]
            if mode == "identity":
                mapping = {
                    key: int(value)
                    for key, value in _mapping_items(
                        entry.get("mapping", []),
                        value_plural="states",
                        value_singular="state",
                    )
                }
                mapped = series.astype("object").map(mapping)
                fallback = int(entry.get("unknown_state", entry.get("missing_state", 0)))
                states = mapped.fillna(fallback).to_numpy(dtype=np.int16)
                representation = entry.get(
                    "representation", self.metadata.get("category_identity", "state")
                )
                if representation == "binary":
                    bits = max(
                        1,
                        int(np.ceil(np.log2(max(int(entry["cardinality"]), 2)))),
                    )
                    output.extend(((states >> bit) & 1)[:, None] for bit in range(bits))
                else:
                    output.append(states[:, None])
            else:
                mapping = {
                    key: float(value)
                    for key, value in _mapping_items(
                        entry.get("mapping", []),
                        value_plural="scores",
                        value_singular="score",
                    )
                }
                scores = (
                    series.astype("object")
                    .map(mapping)
                    .fillna(float(entry["default_score"]))
                    .to_numpy(dtype=np.float64)
                )
                thresholds = self.arrays[entry["thresholds_array"]]
                states = np.searchsorted(thresholds, scores, side="right") + 1
                states[series.isna().to_numpy()] = 0
                output.append(states.astype(np.int16)[:, None])

        if self.metadata.get("embedding") is not None:
            raise ValueError("portable typed adapter does not support embedding IR yet")
        matrix = (
            np.column_stack(output).astype(np.float64, copy=False)
            if output
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=list(self.metadata.get("output_feature_names", [])),
            feature_kinds=list(self.metadata.get("output_feature_kinds", [])),
            cardinalities=list(self.metadata.get("output_cardinalities", [])),
            metadata={
                "adapter_version": self.metadata.get("adapter_version"),
                "portable": True,
            },
        ).validate(len(frame))
