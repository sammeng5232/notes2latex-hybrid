from n2lh.recognition.prompts import PACKAGES, error_hints, fix_user_prompt


def test_hint_for_the_common_ampersand_error():
    hints = error_hints(["Misplaced alignment tab character &. (line 75)"])
    assert len(hints) == 1 and "\\&" in hints[0] and "and" in hints[0]


def test_hint_for_aligned_outside_math_mode():
    hints = error_hints(["Package amsmath Error: \\begin{aligned} allowed only in math mode."])
    assert any("\\[" in h for h in hints)


def test_hint_for_undefined_environment_lists_available_packages():
    hints = error_hints(["LaTeX Error: Environment foo undefined."])
    assert hints and "tikz-cd" in hints[0] and PACKAGES in hints[0]


def test_no_hint_for_an_error_we_have_no_advice_on():
    assert error_hints(["Runaway argument?"]) == []


def test_fix_prompt_carries_errors_latex_and_hints():
    prompt = fix_user_prompt("a & b", ["Misplaced alignment tab character &. (line 7)"])
    assert "a & b" in prompt
    assert "Misplaced alignment tab" in prompt
    assert "Hints:" in prompt and "\\&" in prompt


def test_fix_prompt_without_a_known_error_has_no_hints_section():
    prompt = fix_user_prompt("x", ["something odd"])
    assert "Hints:" not in prompt

def test_hint_for_a_tikzcd_arrow_pointing_at_an_empty_cell():
    hints = error_hints(["Package pgf Error: No shape named `tikz@f@1-2-2' is known. (line 107)"])
    assert hints and "empty" in hints[0] and "array" in hints[0]


def test_first_repair_prompt_does_not_escalate_but_repeat_failures_do():
    plain = fix_user_prompt("x", ["Misplaced alignment tab character &."], repeats=0)
    assert "SAME error" not in plain and "[diagram omitted]" not in plain
    again = fix_user_prompt("x", ["Misplaced alignment tab character &."], repeats=2)
    assert "SAME error" in again and "previous 2 fix attempt(s)" in again
    assert "[diagram omitted]" in again