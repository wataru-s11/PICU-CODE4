import drug_panel


class FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, v):
        self._value = v


class FakeSpinbox:
    def __init__(self, text: str):
        self._text = text

    def get(self) -> str:
        return self._text


def test_get_values_reads_spinbox_when_var_unsynced():
    panel = drug_panel.DrugPanel.__new__(drug_panel.DrugPanel)
    panel.vars = {"adrenaline": FakeVar(0.0)}
    panel.spinboxes = {"adrenaline": FakeSpinbox("0.2")}

    vals = panel.get_values()

    assert vals["adrenaline"] == 0.2
    # internal variable should also be updated
    assert panel.vars["adrenaline"].get() == 0.2

