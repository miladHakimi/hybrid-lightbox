from lightbox.ai import select_ai_device


def test_ai_uses_cpu_by_default_to_avoid_native_metal_command_buffer_crashes():
    class MPS:
        @staticmethod
        def is_available():
            return True

    class Backends:
        mps = MPS()

    class Torch:
        backends = Backends()

    assert select_ai_device(Torch(), platform_name="darwin") == -1
    assert select_ai_device(Torch(), requested="mps", platform_name="darwin") == "mps"
