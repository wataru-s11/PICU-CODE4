import types

import main_surgery as ms


def test_yn_dialog_falls_back_to_console(monkeypatch):
    """When called outside the main thread the dialog should use console I/O."""

    # Simulate a background thread by ensuring ``current_thread`` is different
    # from ``main_thread``.
    monkeypatch.setattr(ms.threading, "current_thread", lambda: object())
    monkeypatch.setattr(ms.threading, "main_thread", lambda: object())

    captured_prompts = []

    def fake_input(prompt=""):
        captured_prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)
    # Suppress console feedback printed by the fallback loop.
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    result = ms.yn_dialog("確認", "テスト入力")

    assert result == "Y"
    assert captured_prompts == ["[確認] テスト入力 (Y/N): "]
