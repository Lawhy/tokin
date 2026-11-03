import numpy as np
import pytest

from tokin.rollout import Generation, Prompt, Rollout, Support, TopLogprobs, Turn

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}


def gen(*ids, **kwargs):
    return Generation(list(ids), "stop", **kwargs)


def turn(prompt, generation, messages=(USER, REPLY)):
    return Turn(Prompt(list(prompt)), generation, list(messages))


def test_generation_rejects_misaligned_logprobs():
    with pytest.raises(ValueError, match="2 entries but token_ids has 3"):
        gen(1, 2, 3, logprobs=[0.0, 0.0])


def test_generation_rejects_a_misaligned_support():
    one = Support(bytes(4), bytes(4), np.array([0, 1], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="covers 1 positions but token_ids has 2"):
        gen(1, 2, support=one)


def test_top_logprobs_of_different_widths_do_not_join():
    with pytest.raises(ValueError, match="of 2 and 3 ids per position"):
        TopLogprobs(bytes(8), bytes(8), 2) + TopLogprobs(bytes(12), bytes(12), 3)


def test_supports_join_with_the_later_offsets_shifted():
    def of(*ids_per_position):
        ids = [i for row in ids_per_position for i in row]
        offsets = np.cumsum([0, *map(len, ids_per_position)]).astype("<i4")
        return Support(np.array(ids, "<i4").tobytes(), np.zeros(len(ids), "<f4").tobytes(), offsets.tobytes())

    assert of([1, 2], [3]) + of([4], [5, 6]) == of([1, 2], [3], [4], [5, 6])


def test_token_ids_concatenate_each_turns_prompt_then_generation():
    r = Rollout([turn([1, 2], gen(3)), turn([4], gen(5, 6))])
    assert r.token_ids == [1, 2, 3, 4, 5, 6]


def test_loss_mask_marks_the_sampled_ids():
    r = Rollout([turn([1, 2], gen(3, 4)), turn([5], gen(6))])
    assert r.loss_mask == [0, 0, 1, 1, 0, 1]


def test_messages_concatenate_each_turns():
    r = Rollout([turn([1], gen(2)), turn([3], gen(4), [{"role": "user", "content": "again"}, REPLY])])
    assert [m["role"] for m in r.messages] == ["user", "assistant", "user", "assistant"]


def test_token_ids_returns_a_fresh_list():
    # Callers concatenate onto it to build the next prompt.
    r = Rollout([turn([1], gen(2))])
    r.token_ids.append(3)
    assert r.token_ids == [1, 2]


def test_repr_summarises_without_dumping_tokens():
    assert repr(Rollout([turn([1, 2, 3], gen(4, 5))])) == "Rollout(tokens=5, generated=2, turns=1)"


def test_logprobs_are_zero_at_rendered_ids():
    r = Rollout([turn([1, 2], gen(3, 4, logprobs=[-0.5, -0.25])), turn([5], gen(6, logprobs=[-1.0]))])
    assert r.logprobs == [0.0, 0.0, -0.5, -0.25, 0.0, -1.0]


def test_logprobs_refuse_a_generation_without_them():
    with pytest.raises(ValueError, match="turn 0 was generated without its logprobs"):
        _ = Rollout([turn([1], gen(2))]).logprobs


def rows(*values):
    return np.array(values, dtype=np.int32).tobytes()


class TestRoutedExperts:
    def test_joins_the_per_turn_slices_in_order(self):
        # Positions 0..3 of 5 tokens, then 4..5 of 7.
        r = Rollout(
            [
                turn([1, 2, 3], gen(4, 5, routed_experts=rows(10, 11, 12, 13))),
                turn([6], gen(7, routed_experts=rows(14, 15))),
            ]
        )
        assert r.routed_experts(layers=1, top_k=1).tolist() == [[[10]], [[11]], [[12]], [[13]], [[14]], [[15]]]

    def test_drops_the_engines_extra_row_for_the_final_token(self):
        r = Rollout([turn([1, 2], gen(3, routed_experts=rows(10, 11, 99)))])
        assert r.routed_experts(layers=1, top_k=1).tolist() == [[[10]], [[11]]]

    def test_refuses_a_slice_of_the_wrong_size(self):
        with pytest.raises(ValueError, match="1 routing rows for 2 positions"):
            Rollout([turn([1, 2], gen(3, routed_experts=rows(10)))]).routed_experts(layers=1, top_k=1)

    def test_refuses_a_generation_without_routing(self):
        with pytest.raises(ValueError, match="without its expert routing"):
            Rollout([turn([1, 2], gen(3))]).routed_experts(layers=1, top_k=1)
