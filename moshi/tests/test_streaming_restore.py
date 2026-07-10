from __future__ import annotations

import torch

from moshi.modules.streaming import RawStreamingConv1d, _flatten_streaming_state


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


if __name__ == "__main__":
    tests = (
        test_restore_none_over_tensor,
        test_restore_tensor_over_none_without_aliasing,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all streaming restore tests passed")
