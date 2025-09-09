import drug_panel

class FakeSpinbox:
    def __init__(self, text: str):
        self._text = text
    def get(self) -> str:
        return self._text

class RaisingVar:
    def __init__(self):
        self.set_calls = []
    def get(self):
        raise Exception("invalid float")
    def set(self, v):
        self.set_calls.append(v)


def test_get_values_handles_invalid_spinbox_text():
    panel = drug_panel.DrugPanel.__new__(drug_panel.DrugPanel)
    panel.vars = {"adrenaline": RaisingVar()}
    panel.spinboxes = {"adrenaline": FakeSpinbox("0..0")}

    vals = panel.get_values()

    assert vals["adrenaline"] == 0.0
    # var.set should be called with fallback value
    assert panel.vars["adrenaline"].set_calls[-1] == 0.0
