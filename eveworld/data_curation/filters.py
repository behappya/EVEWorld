"""Small dataset filters referenced by GigaTrain config files."""

from __future__ import annotations


def select_data_indices(data_list: list[dict], dataset_index: int = 0, indices: list[int] | tuple[int, ...] = ()) -> list[dict]:
    """Select packed-data rows by their stable ``data_index``.

    GigaDatasets applies a filter to the primary label dataset before merging
    video and prompt records, so arbitrary train-only rows can be selected
    without copying the 137 MB packed videos.
    """

    del dataset_index
    wanted = {int(index) for index in indices}
    selected = [row for row in data_list if int(row["data_index"]) in wanted]
    if len(selected) != len(wanted):
        found = {int(row["data_index"]) for row in selected}
        raise ValueError(f"filter requested missing data indices: {sorted(wanted - found)}")
    return selected
