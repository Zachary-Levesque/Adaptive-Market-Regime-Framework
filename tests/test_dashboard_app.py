from streamlit.testing.v1 import AppTest


def test_streamlit_overview_renders_core_status_sections():
    app = AppTest.from_file("dashboard/app.py")
    app.run(timeout=30)

    assert not app.exception

    rendered_text = _rendered_text(app)
    for phrase in [
        "Adaptive Market Regime Framework",
        "Snapshot",
        "Portfolio Allocation",
        "Alpha Sleeve",
        "SPY Sleeve",
        "Sharpe vs SPY",
    ]:
        assert phrase in rendered_text

    assert len(app.metric) >= 8
    assert len(app.dataframe) >= 1


def _rendered_text(app) -> str:
    values: list[str] = []
    for element_name in ["title", "header", "subheader", "markdown", "metric", "warning"]:
        for element in getattr(app, element_name, []):
            for attr in ["label", "value", "body"]:
                if hasattr(element, attr):
                    value = getattr(element, attr)
                    if value is not None:
                        values.append(str(value))
    return "\n".join(values)
