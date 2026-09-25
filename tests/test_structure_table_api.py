from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification

from cerm import CERMClassifier, inspect_model, structure_table


def test_structure_table_combines_main_and_interaction_views():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=29,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = CERMClassifier(
        interaction_order=1,
        resource_policy="ignore",
        random_state=29,
    ).fit(frame, y)

    table = structure_table(model)
    inspection = inspect_model(model)

    assert not table.empty
    assert set(table["kind"]) == {"main"}
    assert set(table["left_feature"]).issubset(set(frame.columns))
    assert inspection.structure.equals(table)
    assert "Model structure" in inspection._repr_html_()
