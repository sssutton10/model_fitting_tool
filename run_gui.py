"""
Launch the Elastic Net Modeling Tool GUI.

Usage:
    panel serve run_gui.py              # preferred
    python run_gui.py                   # auto-opens browser on port 5006
"""

import panel as pn

pn.extension("tabulator", notifications=True)

from elastic_net_tool.gui import create_app

app = create_app()
app.servable()

if __name__ == "__main__":
    pn.serve(
        {"Modeling Tool": lambda: create_app()},
        port=5006,
        show=True,
        title="Elastic Net Modeling Tool",
    )
