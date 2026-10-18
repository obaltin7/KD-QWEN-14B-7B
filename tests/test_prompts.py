from kdq.prompts import (
    build_prompt, clean_solution, format_question, format_target, truncate_at_stop,
)

SHOT = {"question": "Tom has 2 apples and buys 3. How many? ",
        "answer": "2 + 3 = <<2+3=5>>5\n#### 5"}


def test_format_question_exact():
    assert format_question("  How many?  ") == "Question: How many?\nAnswer:"


def test_format_target_leading_space_and_clean():
    assert format_target(SHOT["answer"]) == " 2 + 3 = 5\n#### 5"


def test_clean_solution_multiple_annotations():
    s = "12/60 = $<<12/60=0.2>>0.2\n0.2*50 = $<<0.2*50=10>>10\n#### 10"
    assert clean_solution(s) == "12/60 = $0.2\n0.2*50 = $10\n#### 10"


def test_zero_shot_prompt_exact():
    assert build_prompt("What is 1+1?") == "Question: What is 1+1?\nAnswer:"


def test_k_shot_prompt_exact():
    p = build_prompt("Q?", [SHOT, SHOT])
    one = "Question: Tom has 2 apples and buys 3. How many?\nAnswer: 2 + 3 = 5\n#### 5"
    assert p == f"{one}\n\n{one}\n\nQuestion: Q?\nAnswer:"
    assert "<<" not in p
    assert "<|im_start|>" not in p  # no chat template


def test_prompt_plus_target_is_training_text():
    # The prompt + target concatenation used in training must equal the few-shot example exactly
    assert format_question(SHOT["question"]) + format_target(SHOT["answer"]) == \
        build_prompt(SHOT["question"], [SHOT]).split("\n\n")[0]


def test_truncate_at_stop():
    stops = ["\n\nQuestion:", "\nQuestion:"]
    assert truncate_at_stop(" 5\n#### 5\n\nQuestion: next", stops) == " 5\n#### 5"
    assert truncate_at_stop(" 5\n#### 5\nQuestion: next", stops) == " 5\n#### 5"
    assert truncate_at_stop(" 5\n#### 5", stops) == " 5\n#### 5"
