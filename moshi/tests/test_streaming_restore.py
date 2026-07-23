from __future__ import annotations

from dataclasses import dataclass

import torch

from moshi.modules.streaming import (
    RawStreamingConv1d,
    StreamingModule,
    _flatten_streaming_state,
)


@dataclass
class _RestoreState:
    cache: torch.Tensor
    offset: torch.Tensor

    def reset(self) -> None:
        self.cache.zero_()
        self.offset.zero_()


class _RestoreModule(StreamingModule[_RestoreState]):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def _init_streaming_state(self, batch_size: int) -> _RestoreState:
        return _RestoreState(
            cache=torch.zeros(
                batch_size,
                2,
                3,
                device=self.anchor.device,
                dtype=self.anchor.dtype,
            ),
            offset=torch.zeros(
                batch_size,
                2,
                device=self.anchor.device,
                dtype=torch.int64,
            ),
        )


def restore_error(module, payload: dict) -> str:
    try:
        module.set_streaming_state_inplace(payload)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        return str(error)
    raise AssertionError("Expected streaming restore to fail")


def snapshot(module) -> dict:
    tensors: dict = {}
    metadata: dict = {}
    _flatten_streaming_state(
        tensors, metadata, module.get_streaming_state(), prefix=""
    )
    cloned = {key: value.detach().clone() for key, value in tensors.items()}
    cloned.update(metadata)
    return cloned


def make_conv() -> RawStreamingConv1d:
    conv = RawStreamingConv1d(1, 1, kernel_size=3, bias=False)
    conv.eval()
    return conv


def test_restore_none_over_tensor() -> None:
    conv = make_conv()
    with conv.streaming(1):
        saved = snapshot(conv)
        conv(torch.ones(1, 1, 2))
        assert conv._streaming_state.previous is not None
        conv.set_streaming_state_inplace(dict(saved))
        assert conv._streaming_state.previous is None


def test_restore_tensor_over_none_without_aliasing() -> None:
    conv = make_conv()
    with conv.streaming(1):
        conv(torch.arange(2, dtype=torch.float32).view(1, 1, 2))
        saved = snapshot(conv)
        expected = saved[".previous"].clone()
        conv.reset_streaming()
        assert conv._streaming_state.previous is None

        conv.set_streaming_state_inplace(dict(saved))
        restored = conv._streaming_state.previous
        assert restored is not None
        torch.testing.assert_close(restored, expected)
        restored.add_(10)
        torch.testing.assert_close(saved[".previous"], expected)

        conv.set_streaming_state_inplace(dict(saved))
        torch.testing.assert_close(conv._streaming_state.previous, expected)


def test_restore_rejects_invalid_broadcast_shapes() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        saved = snapshot(module)
        expected = module._streaming_state.cache.clone()
        invalid_values = (
            torch.tensor(1.0),
            torch.zeros(1, 1, 3),
        )

        for invalid_value in invalid_values:
            payload = dict(saved)
            payload[".cache"] = invalid_value
            message = restore_error(module, payload)
            assert ".cache" in message
            assert "expected shape (2, 2, 3)" in message
            assert f"got shape {tuple(invalid_value.shape)}" in message
            assert torch.equal(module._streaming_state.cache, expected)


def test_restore_rejects_dtype_mismatch() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        payload = snapshot(module)
        payload[".cache"] = payload[".cache"].to(torch.float64)

        message = restore_error(module, payload)
        assert ".cache" in message
        assert "expected shape (2, 2, 3) and dtype torch.float32" in message
        assert "got shape (2, 2, 3) and dtype torch.float64" in message


def test_restore_missing_late_key_is_atomic() -> None:
    module = _RestoreModule()
    with module.streaming(2):
        state = module._streaming_state
        state.cache.copy_(torch.arange(12).view(2, 2, 3))
        state.offset.copy_(torch.arange(4).view(2, 2))
        expected_cache = state.cache.clone()
        expected_offset = state.offset.clone()

        payload = snapshot(module)
        payload[".cache"].fill_(99)
        payload.pop(".offset")
        message = restore_error(module, payload)

        assert ".offset" in message
        assert torch.equal(state.cache, expected_cache)
        assert torch.equal(state.offset, expected_offset)


def test_restore_allows_batch_broadcast() -> None:
    module = _RestoreModule()
    with module.streaming(3):
        payload = snapshot(module)
        cache = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
        offset = torch.tensor([[4, 5]], dtype=torch.int64)
        payload[".cache"] = cache
        payload[".offset"] = offset

        module.set_streaming_state_inplace(payload)

        state = module._streaming_state
        assert torch.equal(state.cache, cache.expand_as(state.cache))
        assert torch.equal(state.offset, offset.expand_as(state.offset))


def assert_invalid_scalar_restore_is_atomic(value) -> None:
    module = _RestoreModule()
    with module.streaming(2):
        state = module._streaming_state
        state.cache.copy_(torch.arange(12).view(2, 2, 3))
        state.offset.copy_(torch.arange(4).view(2, 2))
        expected_cache = state.cache.clone()
        expected_offset = state.offset.clone()

        payload = snapshot(module)
        payload[".cache"].fill_(99)
        payload[".offset"] = value
        message = restore_error(module, payload)

        assert ".offset" in message
        assert torch.equal(state.cache, expected_cache)
        assert torch.equal(state.offset, expected_offset)


def test_restore_rejects_string_for_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic("invalid")


def test_restore_rejects_none_for_required_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(None)


def test_restore_rejects_fractional_integer_tensor_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(1.5)


def test_restore_rejects_integer_overflow_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(2**63)


def test_restore_rejects_nan_scalar_atomically() -> None:
    assert_invalid_scalar_restore_is_atomic(float("nan"))


if __name__ == "__main__":
    tests = (
        test_restore_none_over_tensor,
        test_restore_tensor_over_none_without_aliasing,
        test_restore_rejects_invalid_broadcast_shapes,
        test_restore_rejects_dtype_mismatch,
        test_restore_missing_late_key_is_atomic,
        test_restore_allows_batch_broadcast,
        test_restore_rejects_string_for_tensor_atomically,
        test_restore_rejects_none_for_required_tensor_atomically,
        test_restore_rejects_fractional_integer_tensor_atomically,
        test_restore_rejects_integer_overflow_atomically,
        test_restore_rejects_nan_scalar_atomically,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all streaming restore tests passed")
