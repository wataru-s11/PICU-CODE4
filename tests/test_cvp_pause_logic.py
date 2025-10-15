import main_surgery as ms


def test_handle_cvp_check_n_sets_no_pause():
    vitals_memory = {
        "CVP_LINE_CHECK_count": 5,
        "CVP_CHECK_PAUSE_UNTIL": 123,
        "CVP_NEXT_R_TS": 456,
    }
    vitals = {"CVP_NEXT_R_TS": 456}
    ms.handle_cvp_check_n(vitals_memory, vitals)
    assert vitals_memory["CVP_LINE_CHECK_count"] == 0
    assert vitals_memory["CVP_CHECK_PAUSE_UNTIL"] is None
    assert vitals_memory["CVP_NEXT_R_TS"] is None
    assert vitals["CVP_NEXT_R_TS"] is None


def test_handle_cvp_observation_comment_after_three():
    vm = {"CVP_OBS_COUNT": 0}
    thresholds = {"CVP_u": 5, "SBP_l": 70, "SBP_u": 90}
    vitals = {"CVP": 6, "SBP": 80}
    assert ms.handle_cvp_observation_comment(vm, vitals, thresholds) == ""
    assert ms.handle_cvp_observation_comment(vm, vitals, thresholds) == ""
    comment = ms.handle_cvp_observation_comment(vm, vitals, thresholds)
    assert "CVP_u基準値を変更してください" in comment
    assert vm["CVP_OBS_COUNT"] == 0


def test_handle_cvp_observation_comment_resets_on_miss():
    vm = {"CVP_OBS_COUNT": 0}
    thresholds = {"CVP_u": 5, "SBP_l": 70, "SBP_u": 90}
    vitals_ok = {"CVP": 6, "SBP": 80}
    vitals_ng = {"CVP": 6, "SBP": 95}

    assert ms.handle_cvp_observation_comment(vm, vitals_ok, thresholds) == ""
    assert vm["CVP_OBS_COUNT"] == 1

    assert ms.handle_cvp_observation_comment(vm, vitals_ng, thresholds) == ""
    assert vm["CVP_OBS_COUNT"] == 0

    assert ms.handle_cvp_observation_comment(vm, vitals_ok, thresholds) == ""
    assert vm["CVP_OBS_COUNT"] == 1
