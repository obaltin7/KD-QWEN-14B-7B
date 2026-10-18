import pytest

from kdq.config import load_config
from kdq.data import content_sha256, make_split_indices, pick_fewshot
from kdq.prompts import build_prompt, clean_solution


def test_split_sizes_and_disjoint():
    val, rest = make_split_indices(7473, 300, 42)
    assert len(val) == 300 and len(rest) == 7173
    assert not set(val) & set(rest)
    assert sorted(val + rest) == list(range(7473))


def test_split_deterministic():
    assert make_split_indices(7473, 300, 42) == make_split_indices(7473, 300, 42)
    assert make_split_indices(7473, 300, 42)[0] != make_split_indices(7473, 300, 1234)[0]


def test_split_invalid():
    with pytest.raises(ValueError):
        make_split_indices(10, 10, 0)


def test_fewshot_from_pool():
    _, rest = make_split_indices(7473, 300, 42)
    shots = pick_fewshot(rest, 4, 42)
    assert len(set(shots)) == 4 and set(shots) <= set(rest)
    assert shots == pick_fewshot(rest, 4, 42)


def test_content_sha256_order_sensitive():
    a = {"question": "q1", "answer": "a1"}
    b = {"question": "q2", "answer": "a2"}
    assert content_sha256([a, b]) != content_sha256([b, a])
    assert content_sha256([a, b]) == content_sha256([dict(a), dict(b)])


def test_prompt_format():
    assert clean_solution("48/2 = <<48/2=24>>24\n#### 24") == "48/2 = 24\n#### 24"
    p = build_prompt("How many?", [{"question": "Q1?", "answer": "1+1=<<1+1=2>>2\n#### 2"}])
    assert p.endswith("Question: How many?\nAnswer:")
    assert "<<" not in p and "#### 2" in p


def test_base_config_values():
    cfg = load_config()
    assert cfg["models"]["teacher"] == "Qwen/Qwen2.5-14B-Instruct"
    assert cfg["models"]["student_base"] == "Qwen/Qwen2.5-7B"
    assert cfg["split"] == {"val_size": 300, "seed": 42, "fewshot_k": 4, "fewshot_seed": 42}
    assert cfg["eval"]["max_new_tokens"] == 512 and cfg["eval"]["shots"] == [0, 4]
    assert cfg["train"]["seeds"] == [42, 1234]
    t = cfg["train"]
    assert t["hidden"]["pairs"] == [[7, 12], [14, 24], [21, 36], [28, 48]]
    assert t["lora"]["r"] == 16 and t["lora"]["alpha"] == 32 and t["kd"] == {
        "tau": 2.0, "alpha": 0.5, "kl_direction": "forward", "vocab_cut": 151665}
    assert t["hidden"]["beta"] == 1.0 and t["effective_batch"] == 16 and t["epochs"] == 2
