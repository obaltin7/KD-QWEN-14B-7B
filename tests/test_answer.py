import pytest

from kdq.answer import extract_gold, extract_pred, is_correct


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72", "72"),
        ("... total\n#### 1,234", "1234"),
        ("#### 1,000,000", "1000000"),
        ("#### -5", "-5"),
        ("#### 3.50", "3.5"),
        ("#### 0", "0"),
    ],
)
def test_extract_gold(answer, expected):
    assert extract_gold(answer) == expected


def test_extract_gold_missing_marker():
    with pytest.raises(ValueError):
        extract_gold("The answer is 5")


@pytest.mark.parametrize(
    "text, expected",
    [
        # text after "####" takes precedence
        ("She has 3 apples and buys 4 more.\n#### 7", "7"),
        ("Step 1: 10 + 5 = 15\n#### 15.\nExtra text 99", "15"),
        ("#### $1,250", "1250"),
        ("####-12", "-12"),
        # without "####": last number
        ("So the total is 18.", "18"),
        ("The answer is $2,500.", "2500"),
        ("He loses 3 dollars, ending with -3.", "-3"),
        ("The cost is 4.75 dollars.", "4.75"),
        ("The result is 5.0.", "5"),
        ("Answer: 1,234,567", "1234567"),
        ("It is 50% of 20, so 10", "10"),
        ("Total = $ 45", "45"),
        ("Range 3-5 gives 5", "5"),
        # "####" present but no number after it -> last number in the text
        ("We get 42 in total.\n####", "42"),
        # multiple "####" -> the last one
        ("#### 3\nWait, recompute.\n#### 4", "4"),
        # no number
        ("I don't know.", None),
        ("", None),
    ],
)
def test_extract_pred(text, expected):
    assert extract_pred(text) == expected


def test_extract_pred_none_input():
    assert extract_pred(None) is None


@pytest.mark.parametrize(
    "pred, gold, ok",
    [
        ("72", "72", True),
        ("72.0", "72", True),
        ("72.00", "72", True),
        ("-5", "-5", True),
        ("5", "-5", False),
        ("71", "72", False),
        ("0.5", "0.50", True),
        (None, "72", False),
        ("abc", "72", False),
    ],
)
def test_is_correct(pred, gold, ok):
    assert is_correct(pred, gold) is ok


def test_end_to_end():
    gold = extract_gold("Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n#### 10")
    assert is_correct(extract_pred("She earned $10.00 in total."), gold)
    assert not is_correct(extract_pred("She earned $0.2 per minute."), gold)


# ---------------------------------------------------------------------------
# Answer extraction v2: strict + flexible
# ---------------------------------------------------------------------------
from kdq.answer import extract_flexible, extract_strict, truncate_generation  # noqa: E402

# Real example from the pod cross-check (idx 0, gold 18)
REAL_IDX0 = (" Janet has 16 - 3 - 4 = 9 eggs left over. She makes 9 * 2 = $18 per day at the "
             "farmers' market. The answer is 18.\n[Question]A store sells 20 packets of 100 grams "
             "of sugar every week. How many kilograms of sugar does it sell every week? "
             "The answer is 2.\nQ: A school has 40 students")
REAL_Q = (" It takes 2 / 2 = 1 bolt of white fiber. So the total is 2 + 1 = 3 bolts. "
          "The answer is 3.\nQ: Josh decides to try flipping a house. He buys it for $80,000. "
          "The answer is 70000.\nA: 70000")
REAL_HASH = (" 3 + 4 = 7\n#### 7\nQuestion: Another one with 99 apples?\nAnswer: 99")


@pytest.mark.parametrize(
    "text, gold",
    [(REAL_IDX0, "18"), (REAL_Q, "3"), (REAL_HASH, "7")],
)
def test_flexible_real_examples(text, gold):
    pred, cut = extract_flexible(text)
    assert pred == gold and cut is not None
    assert is_correct(pred, gold)


def test_v1_fails_on_real_example():
    # v1 (last number) picked the number from hallucinated questions
    assert extract_pred(REAL_IDX0) != "18"


@pytest.mark.parametrize(
    "text, expected_cut",
    [
        ("A\n[Question] B", "A"),
        ("A\nQ: B", "A"),
        ("A\nq1: B", "A"),
        ("A\nQuestion 2: B", "A"),
        ("A\n  problem: B", "A"),
        ("A\n### Question\nB", "A"),
        ("A\n[question 3] B", "A"),
        ("A\n#### 5\nextra 9", "A\n#### 5"),
        ("Questions are fun: 5", "Questions are fun: 5"),     # first line is never cut
        ("A\nQuestionable claim 4", "A\nQuestionable claim 4"),  # word boundary
        ("A\nThe question is 4", "A\nThe question is 4"),        # not at line start
    ],
)
def test_truncate_generation(text, expected_cut):
    assert truncate_generation(text)[0] == expected_cut


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so\n#### 72", "72"),
        ("#### 3\nlater #### 4", "3"),          # first occurrence
        ("####   $1,234.50", "1234.5"),
        ("#### -7", "-7"),
        ("#### The answer is 5", None),        # no number right after ####
        ("The answer is 5.", None),             # no ####
        ("", None),
    ],
)
def test_extract_strict(text, expected):
    assert extract_strict(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (" 10 + 8 = 18\n#### 18\nQ: 99", "18"),
        (" The answer is $1,234.", "1234"),
        (" Total 5 then 6. The answer is: 6.", "6"),
        (" The ANSWER IS -3.5 degrees, not 4.", "-3.5"),   # answer is > last number
        (" The answer is 12. Check: 12 + 1 = 13", "12"),     # first 'answer is'
        (" The answer is clear. So 8.", "8"),                # no number after 'answer is' -> last number
        (" She has 0.50 left.", "0.5"),
        (" It drops to -4.\n[Question] 77", "-4"),
        (" no numbers here", None),
    ],
)
def test_extract_flexible(text, expected):
    assert extract_flexible(text)[0] == expected


def test_strict_vs_flexible_differ():
    text = " 9 * 2 = 18. The answer is 18."
    assert extract_strict(text) is None
    assert extract_flexible(text)[0] == "18"


# ---------------------------------------------------------------------------
# Robustness: very long digit strings (student_base 4-shot crash on the pod)
# ---------------------------------------------------------------------------
from kdq.answer import MAX_DIGITS  # noqa: E402

LONG = "9" * 200
OK30 = "1" + "0" * (MAX_DIGITS - 1)          # exactly 30 digits: valid


@pytest.mark.parametrize(
    "text",
    [
        f"#### {LONG}",
        f"So it is {LONG}",
        f"#### {LONG}.5",
        f"The answer is {'1,000' * 20}",
        f"0.{LONG}",
        f"x\n#### 5\n#### {LONG}",
    ],
)
def test_long_numbers_do_not_crash(text):
    # v1 (the path that crashed on the pod) and v2 must not raise
    extract_pred(text)
    extract_strict(text)
    extract_flexible(text)


def test_long_number_is_unanswered():
    assert extract_pred(f"#### {LONG}") is None
    assert extract_strict(f"#### {LONG}") is None
    assert extract_flexible(f"#### {LONG}")[0] is None
    # an invalid number does not fall through to another number: no answer at that step
    assert extract_flexible(f"Got 18 then. The answer is {LONG}.")[0] is None
    assert extract_flexible(f"Got 18, then {LONG}")[0] is None


def test_max_digits_boundary():
    assert extract_strict(f"#### {OK30}") == OK30
    assert extract_strict(f"#### {OK30}0") is None
    assert is_correct(extract_strict(f"#### {OK30}"), OK30)


def test_long_number_never_correct():
    assert not is_correct(extract_flexible(f"#### {LONG}")[0], "18")
