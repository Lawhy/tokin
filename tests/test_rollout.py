import numpy as np
import pytest

from tokin.rollout import Generation, Prompt, Rollout, Session


def gen(*ids, **kwargs):
    return Generation(list(ids), "stop", **kwargs)


def test_generation_rejects_misaligned_logprobs():
    with pytest.raises(ValueError, match="2 entries but token_ids has 3"):
        gen(1, 2, 3, logprobs=[0.0, 0.0])


def test_token_ids_concatenate_in_segment_order():
    r = Rollout()
    r.append(Prompt([1]), [])
    for i in range(3):
        r.append(gen(100 + i), [])
        r.append(Prompt([200 + i]), [])
    assert r.token_ids == [1, 100, 200, 101, 201, 102, 202]


def test_segments_keep_their_kind():
    r = Rollout()
    r.append(Prompt([1]), [])
    r.append(gen(2), [])
    assert [type(s) for s in r.segments] == [Prompt, Generation]


def test_empty_segments_are_dropped():
    r = Rollout()
    r.append(Prompt([1]), [])
    r.append(Prompt([]), [])
    r.append(gen(), [])
    assert len(r.segments) == 1


def test_token_ids_returns_a_fresh_list():
    # Callers concatenate onto it to build the next prompt.
    r = Rollout()
    r.append(Prompt([1]), [])
    r.token_ids.append(2)
    assert r.token_ids == [1]


def test_repr_summarises_without_dumping_tokens():
    r = Rollout()
    r.append(Prompt([1, 2, 3]), [])
    r.append(gen(4, 5), [{"role": "assistant", "content": "hi"}])
    assert repr(r) == "Rollout(tokens=5, generated=2, segments=2, messages=1)"


def test_segments_carry_their_messages():
    r = Rollout()
    r.append(Prompt([1, 2]), [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    r.append(gen(), [{"role": "assistant", "content": ""}])  # no tokens, but the harness still got a reply
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant"] and len(r.segments) == 1


def test_a_generation_cannot_open_a_rollout():
    # A loss mask starting at 1 would train on tokens nothing conditioned on.
    with pytest.raises(RuntimeError, match="cannot open with a generation"):
        Rollout().append(gen(1), [])


def test_loss_mask_marks_the_sampled_ids():
    r = Rollout()
    r.append(Prompt([1, 2]), [])
    r.append(gen(3, 4), [])
    r.append(Prompt([5]), [])
    r.append(gen(6), [])
    assert r.loss_mask == [0, 0, 1, 1, 0, 1]


def test_logprobs_align_with_token_ids():
    r = Rollout()
    r.append(Prompt([1, 2]), [])
    r.append(gen(3, 4, logprobs=[-0.5, -0.25]), [])
    r.append(Prompt([5]), [])
    r.append(gen(6, logprobs=[-1.0]), [])
    assert r.logprobs == [0.0, 0.0, -0.5, -0.25, 0.0, -1.0]


def test_logprobs_refuse_a_generation_without_them():
    r = Rollout()
    r.append(Prompt([1]), [])
    r.append(gen(2), [])
    with pytest.raises(ValueError, match="segment 1 was generated without its logprobs"):
        _ = r.logprobs


def rows(*values):
    return np.array(values, dtype=np.int32).tobytes()


class TestRoutedExperts:
    def test_joins_the_per_call_slices_in_order(self):
        r = Rollout()
        r.append(Prompt([1, 2, 3]), [])
        r.append(gen(4, 5, routed_experts=rows(10, 11, 12, 13)), [])  # positions 0..3 of 5 tokens
        r.append(Prompt([6]), [])
        r.append(gen(7, routed_experts=rows(14, 15)), [])  # positions 4..5 of 7 tokens
        assert r.routed_experts(layers=1, top_k=1).tolist() == [[[10]], [[11]], [[12]], [[13]], [[14]], [[15]]]

    def test_drops_the_engines_extra_row_for_the_final_token(self):
        r = Rollout()
        r.append(Prompt([1, 2]), [])
        r.append(gen(3, routed_experts=rows(10, 11, 99)), [])
        assert r.routed_experts(layers=1, top_k=1).tolist() == [[[10]], [[11]]]

    def test_refuses_a_slice_of_the_wrong_size(self):
        r = Rollout()
        r.append(Prompt([1, 2]), [])
        r.append(gen(3, routed_experts=rows(10)), [])
        with pytest.raises(ValueError, match="1 routing rows for 2 positions"):
            r.routed_experts(layers=1, top_k=1)

    def test_refuses_a_generation_without_routing(self):
        r = Rollout()
        r.append(Prompt([1, 2]), [])
        r.append(gen(3), [])
        with pytest.raises(ValueError, match="without its expert routing"):
            r.routed_experts(layers=1, top_k=1)


def test_session_starts_with_one_empty_rollout():
    s = Session()
    assert len(s.rollouts) == 1
    assert len(s.current) == 0


def test_session_turns_append_to_its_current_rollout():
    s = Session()
    s.current.append(Prompt([1, 2]), [])
    s.current.append(Generation([3], "stop", logprobs=[0.0]), [])
    assert len(s.rollouts) == 1
    assert s.current.token_ids == [1, 2, 3]


def test_session_repr_summarises_without_dumping_tokens():
    s = Session()
    s.current.append(Prompt([1, 2, 3]), [])
    assert repr(s) == f"Session(id={s.id!r}, rollouts=1, tokens=3)"


def test_sessions_do_not_share_their_default_rollout():
    a, b = Session(), Session()
    a.current.append(Prompt([1]), [])
    assert len(b.current) == 0


def test_sessions_do_not_share_a_lock():
    assert Session().lock is not Session().lock


def test_session_id_is_fresh_unless_given():
    assert Session().id != Session().id and len(Session().id) == 32
    assert Session(id="episode-7").id == "episode-7"
