"""
Codec profiles and parameters for identification
Based on the research paper methodology adapted for network codecs
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

class TransformType(Enum):
    MDCT = "mdct"
    MDST = "mdst"
    DCT = "dct"
    FFT = "fft"
    LP = "linear_prediction"
    ACELP = "acelp"
    SUBBAND = "subband"
    PCM = "pcm"
    ADPCM = "adpcm"
    HYBRID = "hybrid"

class WindowType(Enum):
    SINE = "sine"
    KBD = "kbd"
    SLOPE = "slope"
    RECTANGULAR = "rectangular"
    HAMMING = "hamming"
    HANNING = "hanning"

@dataclass
class CodecProfile:
    name: str
    sample_rates: List[int]
    bitrates: List[int]
    frame_sizes: List[int]  # in samples
    transform_type: TransformType
    window_type: WindowType
    window_params: Dict[str, Any]
    hop_size_ratio: float  # hop_size = frame_size * hop_size_ratio
    frequency_bands: List[Tuple[int, int]]  # frequency band ranges
    additional_params: Dict[str, Any]

# Define codec profiles based on specifications
CODEC_PROFILES = {
    # Music/General Audio Codecs (from paper)
    "MP3": CodecProfile(
        name="MP3",
        sample_rates=[44100, 48000],
        bitrates=[96000, 128000, 192000, 256000, 320000],
        frame_sizes=[1152, 384],  # long and short windows
        transform_type=TransformType.HYBRID,
        window_type=WindowType.SINE,
        window_params={},
        hop_size_ratio=0.5,
        frequency_bands=[(0, 22050)],
        additional_params={"subband_samples": 32, "mdct_lines": 18}
    ),
    
    "AAC": CodecProfile(
        name="AAC",
        sample_rates=[44100, 48000],
        bitrates=[96000, 128000, 192000, 256000, 320000],
        frame_sizes=[2048, 256],  # long and short windows
        transform_type=TransformType.MDCT,
        window_type=WindowType.KBD,
        window_params={"alpha": 4},  # alpha=6 for short window
        hop_size_ratio=0.5,
        frequency_bands=[(0, 22050)],
        additional_params={"scalefactor_bands": 49}
    ),
    
    "AC-3": CodecProfile(
        name="AC-3",
        sample_rates=[48000],
        bitrates=[192000, 384000, 448000],
        frame_sizes=[512, 256],
        transform_type=TransformType.MDCT,
        window_type=WindowType.KBD,
        window_params={"alpha": 5},
        hop_size_ratio=0.5,
        frequency_bands=[(0, 24000)],
        additional_params={"exponent_strategy": True}
    ),
    
    "Vorbis": CodecProfile(
        name="Vorbis",
        sample_rates=[44100, 48000],
        bitrates=[96000, 128000, 192000, 256000, 320000],
        frame_sizes=[2048, 4096],  # variable, powers of 2 from 64 to 8192
        transform_type=TransformType.MDCT,
        window_type=WindowType.SLOPE,
        window_params={},
        hop_size_ratio=0.5,
        frequency_bands=[(0, 22050)],
        additional_params={"floor_type": 1}
    ),
    
    # Network/Telephony Codecs
    "G.711": CodecProfile(
        name="G.711",
        sample_rates=[8000],
        bitrates=[64000],
        frame_sizes=[1],  # sample-by-sample
        transform_type=TransformType.PCM,
        window_type=WindowType.RECTANGULAR,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(300, 3400)],
        additional_params={"companding": ["ulaw", "alaw"]}
    ),
    
    "G.722": CodecProfile(
        name="G.722",
        sample_rates=[16000],
        bitrates=[48000, 56000, 64000],
        frame_sizes=[2],  # processes 2 samples at a time
        transform_type=TransformType.SUBBAND,
        window_type=WindowType.RECTANGULAR,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(0, 4000), (4000, 8000)],
        additional_params={"qmf_taps": 24, "adpcm_bits": [6, 2]}
    ),
    
    "AMR-NB": CodecProfile(
        name="AMR-NB",
        sample_rates=[8000],
        bitrates=[4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200],
        frame_sizes=[160],  # 20ms frames
        transform_type=TransformType.ACELP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(300, 3400)],
        additional_params={"lp_order": 10, "subframes": 4}
    ),
    
    "AMR-WB": CodecProfile(
        name="AMR-WB",
        sample_rates=[16000],
        bitrates=[6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850],
        frame_sizes=[320],  # 20ms frames
        transform_type=TransformType.ACELP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(50, 7000)],
        additional_params={"lp_order": 16, "subframes": 4}
    ),
    
    "EVS": CodecProfile(
        name="EVS",
        sample_rates=[8000, 16000, 32000, 48000],
        bitrates=[5900, 7200, 8000, 9600, 13200, 16400, 24400, 32000, 48000, 64000, 96000, 128000],
        frame_sizes=[160, 320, 640, 960],  # 20ms frames at different sample rates
        transform_type=TransformType.HYBRID,
        window_type=WindowType.SINE,
        window_params={},
        hop_size_ratio=0.5,
        frequency_bands=[(20, 20000)],
        additional_params={"modes": ["ACELP", "MDCT", "TCX"]}
    ),
    
    "Opus": CodecProfile(
        name="Opus",
        sample_rates=[8000, 12000, 16000, 24000, 48000],
        bitrates=[6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 256000],
        frame_sizes=[120, 240, 480, 960, 1920, 2880],  # 2.5-60ms frames
        transform_type=TransformType.HYBRID,
        window_type=WindowType.SINE,
        window_params={},
        hop_size_ratio=0.5,
        frequency_bands=[(20, 20000)],
        additional_params={"modes": ["SILK", "CELT", "Hybrid"], "complexity": 10}
    ),
    
    "Speex": CodecProfile(
        name="Speex",
        sample_rates=[8000, 16000, 32000],
        bitrates=[2150, 5950, 8000, 11000, 15000, 18200, 24600, 42200],
        frame_sizes=[160, 320, 640],  # 20ms frames
        transform_type=TransformType.ACELP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(0, 4000), (0, 8000)],
        additional_params={"lp_order": 10, "subframes": 4}
    ),
    
    "iLBC": CodecProfile(
        name="iLBC",
        sample_rates=[8000],
        bitrates=[13330, 15200],
        frame_sizes=[160, 240],  # 20ms or 30ms frames
        transform_type=TransformType.LP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(300, 3400)],
        additional_params={"lpc_order": 10, "block_independent": True}
    ),
    
    "GSM-FR": CodecProfile(
        name="GSM-FR",
        sample_rates=[8000],
        bitrates=[13000],
        frame_sizes=[160],  # 20ms frames
        transform_type=TransformType.LP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(300, 3400)],
        additional_params={"lpc_order": 8, "rpe": True}
    ),
    
    "SILK": CodecProfile(
        name="SILK",
        sample_rates=[8000, 12000, 16000, 24000],
        bitrates=[6000, 8000, 12000, 20000, 40000],
        frame_sizes=[160, 240, 320, 480],  # 20ms frames
        transform_type=TransformType.LP,
        window_type=WindowType.HAMMING,
        window_params={},
        hop_size_ratio=1.0,
        frequency_bands=[(0, 4000), (0, 8000)],
        additional_params={"lpc_order": 16, "noise_shaping": True}
    ),
}

def get_codec_profile(codec_name: str) -> CodecProfile:
    """Get codec profile by name"""
    return CODEC_PROFILES.get(codec_name.upper())

def get_all_codec_names() -> List[str]:
    """Get list of all supported codec names"""
    return list(CODEC_PROFILES.keys())

def get_codec_by_bitrate(bitrate: int) -> List[str]:
    """Get codecs that support a specific bitrate"""
    matching_codecs = []
    for name, profile in CODEC_PROFILES.items():
        if bitrate in profile.bitrates:
            matching_codecs.append(name)
    return matching_codecs