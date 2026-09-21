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

# ----------------------------------------------- transcribe, do not normalise
def test_the_transcribe_prompt_forbids_re_parameterising_the_mathematics():
    """A real page wrote Exp(theta) with mean theta; the model returned Exp(lambda)
    with mean 1/lambda, silently converting the notes to the convention it knows.
    The output looks right and compiles, so only the prompt can prevent it."""
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    body = TRANSCRIBE_SYSTEM.lower()
    assert "do not normalise" in body
    assert "exp(theta)" in body and "1/lambda" in body      # the worked example
    assert "do not correct, complete or re-derive" in body


def test_the_repair_prompt_also_forbids_it():
    """The repair pass sees the page image too and is just as able to 'improve' it."""
    from n2lh.recognition.prompts import FIX_SYSTEM
    body = FIX_SYSTEM.lower()
    assert "fix only what the compiler complained about" in body
    assert "exp(theta)" in body


def test_the_quantifier_glossary_is_still_there():
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    assert "\\forall" in TRANSCRIBE_SYSTEM and "upside-down A" in TRANSCRIBE_SYSTEM


# ------------------------------------------------------- colored ink and layout
def test_the_prompt_teaches_colored_ink():
    """Real failure: red strikes and blue underlines were on the page, but the
    model was never told colors are content, so none reached the output."""
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    assert "\\textcolor" in TRANSCRIBE_SYSTEM
    assert "colored ink" in TRANSCRIBE_SYSTEM.lower()
    assert "never drop one that is" in TRANSCRIBE_SYSTEM.lower()


def test_the_prompt_teaches_margin_marks_and_strikethrough():
    """Real failure: a margin column of circled section numbers with 极/难/可/必
    marks was dropped entirely, and red-struck marks with it."""
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    body = TRANSCRIBE_SYSTEM.lower()
    assert "margin" in body and "never drop margin marks" in body
    assert "\\cancel" in TRANSCRIBE_SYSTEM


def test_the_prompt_forbids_centering_a_corner_table():
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    assert "corner of the page" in TRANSCRIBE_SYSTEM


def test_doc_hint_block_names_the_files():
    from n2lh.recognition.prompts import doc_hint_block
    hint = doc_hint_block(["实变函数_周民强_总结.pdf"])
    assert "实变函数_周民强_总结.pdf" in hint
    assert "file name" in hint
    assert doc_hint_block([]) == ""


def test_doc_hint_block_caps_the_number_of_files():
    from n2lh.recognition.prompts import doc_hint_block
    hint = doc_hint_block([f"notes-{i}.pdf" for i in range(9)])
    assert "notes-5.pdf" not in hint and "notes-4.pdf" in hint


def test_the_user_prompt_names_the_detected_colors():
    """A generic 'mind the colors' line in the system prompt was ignored; the
    per-page directive must name what image analysis actually found."""
    from n2lh.recognition.prompts import transcribe_user_prompt
    prompt = transcribe_user_prompt("", [], color_ink=["pink", "green"])
    assert "pink and green" in prompt
    assert "\\textcolor" in prompt and "\\cancel" in prompt
    assert "COLORED INK IS PRESENT" in prompt


def test_the_user_prompt_is_silent_without_colors():
    from n2lh.recognition.prompts import transcribe_user_prompt
    assert "COLORED INK" not in transcribe_user_prompt("", [], color_ink=[])


def test_the_prompt_keeps_corner_tables_and_forbids_invented_margin_numbers():
    from n2lh.recognition.prompts import TRANSCRIBE_SYSTEM
    assert "CORNER or the margin" in TRANSCRIBE_SYSTEM
    assert "do not continue the numbering pattern" in TRANSCRIBE_SYSTEM