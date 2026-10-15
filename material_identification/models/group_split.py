"""Групповое разбиение выборки по значению показателя m.

Все объекты с одинаковым m получены масштабированием одной опорной кривой
(свойство подобия решений для закона Бэкофена), т.е. отличаются лишь
постоянным множителем. При случайном разбиении по объектам такие
пропорциональные кривые попадают одновременно в обучающую и в тестовую
части, и оценка точности получается завышенной. Поэтому объекты с одним
значением m целиком относятся либо к обучающей, либо к тестовой части,
а кросс-валидация выполняется по тем же группам (GroupKFold).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

GROUP_COL = "m"


def groups_from(table) -> np.ndarray:
    """Метки групп: значение m, округлённое до 1e-6."""
    return np.round(np.asarray(table[GROUP_COL], dtype=float), 6)


def group_split(*arrays, groups, test_size: float, random_state: int):
    """Аналог sklearn.train_test_split, но без пересечения групп между частями."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=groups))
    out = []
    for a in arrays:
        if hasattr(a, "iloc"):
            out += [a.iloc[tr], a.iloc[te]]
        else:
            a = np.asarray(a)
            out += [a[tr], a[te]]
    return out


def group_kfold(n_splits: int = 5) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)
