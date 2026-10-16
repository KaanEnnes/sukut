"""DeepFilterNet 0.5.6, yeni torchaudio'da kaldirilan torchaudio.backend'i ariyor; kucuk bir yama."""
import sys, types
try:
    import torchaudio.backend.common  # noqa
except Exception:
    from dataclasses import dataclass
    @dataclass
    class AudioMetaData:
        sample_rate: int = 0; num_frames: int = 0; num_channels: int = 0
        bits_per_sample: int = 0; encoding: str = ""
    b = types.ModuleType("torchaudio.backend"); c = types.ModuleType("torchaudio.backend.common")
    c.AudioMetaData = AudioMetaData; b.common = c
    sys.modules["torchaudio.backend"] = b; sys.modules["torchaudio.backend.common"] = c

# Paketlenmis exe'de git cagrisi patlar; DF'nin commit hash'i sadece log icin.
try:
    import df.utils as _u
    _u.get_git_root = lambda: None
    _u.get_commit_hash = lambda: None
except Exception:
    pass
