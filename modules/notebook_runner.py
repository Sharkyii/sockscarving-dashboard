import os

import nbformat
import streamlit as st


def run_notebook(path: str, namespace: dict) -> None:
    """Execute every code cell of a Jupyter notebook in-order inside `namespace`.

    Lets a tab's charts live as editable notebook cells: edit + save the
    notebook, refresh the dashboard, and the new cell code runs immediately.
    Errors in a single cell are shown inline (with the cell number) instead
    of crashing the whole page, so one bad edit doesn't take down the tab.
    """
    if not os.path.exists(path):
        st.warning(f"Notebook not found: {path}")
        return

    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{path}:cell{i}", "exec"), namespace)
        except Exception as e:
            st.error(f"Error in {os.path.basename(path)} (cell {i}): {e}")
