import main_surgery as ms


def test_update_furosemide_flags_sets_pending_for_new_entry():
    vitals = {"timestamp": "2024-01-01 00:00:00", "furosemide_mg": "5"}
    memory = {"FRO_LAST_DOSE_ENTRY": None, "FRO_DOSE_LOGGED": False}

    ms.update_furosemide_flags(vitals, memory)

    assert memory["FRO_DOSE_LOGGED"] is True
    assert memory["FRO_LAST_DOSE_ENTRY"] == ("2024-01-01 00:00:00", 5.0)


def test_update_furosemide_flags_does_not_retrigger_same_entry():
    vitals = {"timestamp": "2024-01-01 00:00:00", "furosemide_mg": 5}
    memory = {"FRO_LAST_DOSE_ENTRY": ("2024-01-01 00:00:00", 5.0), "FRO_DOSE_LOGGED": False}

    ms.update_furosemide_flags(vitals, memory)

    assert memory["FRO_DOSE_LOGGED"] is False


def test_update_furosemide_flags_ignores_invalid_values():
    vitals = {"timestamp": "2024-01-01 00:00:00", "furosemide_mg": ""}
    memory = {"FRO_LAST_DOSE_ENTRY": None, "FRO_DOSE_LOGGED": False}

    ms.update_furosemide_flags(vitals, memory)

    assert memory["FRO_DOSE_LOGGED"] is False
