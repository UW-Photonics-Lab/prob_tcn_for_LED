import math
import subprocess
from pathlib import Path

import pytest
import torch

from experiments.inference_sv import read_model
from modules.models import QxxTCN
from modules.utils import float_to_qxy_int_code, qxy_hex_to_float, qxy_int_to_hex, save_mem_file

LAYERS = 2
KERNEL_SIZE = 3
DILATION_BASE = 2
HIDDEN_CHANNELS = 8
DATA_WIDTH = 8
FRAC_BITS = 3
TEST = 1
SAMPLES = 1000  # must be a multiple of the tb THROUGHPUT
WARMUP_ZEROS = 64  # zeros run through the Python model first so its layer buffers match the SV after reset

SV_DIR = Path(__file__).resolve().parents[1] / "sv_tcn" / "tcn7"


def to_code(x: float) -> int:
    return float_to_qxy_int_code(x, DATA_WIDTH, FRAC_BITS)


def hex_to_code(hexa: str) -> int:
    v = int(hexa, 16)
    return v - (1 << DATA_WIDTH) if v & (1 << (DATA_WIDTH - 1)) else v


def read_mem_codes(path) -> list[int]:
    codes = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            codes.append(hex_to_code(line))
    return codes


@pytest.mark.parametrize("strides", [[1, 1], [2, 1], [1, 2], [2, 2]])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sv_matches_python(strides, seed, tmp_path):
    torch.manual_seed(seed)
    quantization = {"frac_bits": FRAC_BITS, "data_width": DATA_WIDTH}
    model = QxxTCN(
        nlayers=LAYERS,
        dilation_base=DILATION_BASE,
        kernel_size=KERNEL_SIZE,
        hidden_channels=HIDDEN_CHANNELS,
        activation="relu",
        quantization=quantization,
        stride=strides,
    ).eval()

    # snap weights to the fixed-point grid
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(model.tcn[0].quantize(p))

    # export weights where hidden_channel_block reads them: TestingData/Test{TEST}/encoder/...
    torch.save({"encoder": model.state_dict(), "decoder": model.state_dict()}, tmp_path / "model.pt")
    read_model(tmp_path / "model.pt", tmp_path / "TestingData" / f"Test{TEST}", DATA_WIDTH, FRAC_BITS)

    # random input on the grid
    codes_in = [to_code(v) for v in torch.normal(0, 1, size=(SAMPLES,)).tolist()]
    save_mem_file(tmp_path / "input_time_series.mem", [qxy_int_to_hex(c, DATA_WIDTH) for c in codes_in],
                  "pytorch test reference input", printDebug=False)

    packed_strides = sum(s << (5 * i) for i, s in enumerate(strides))
    subprocess.run(
        ["iverilog", "-g2012", "-I", str(SV_DIR), "-P", 'tb.MODEL_TYPE="encoder"',
         "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-P", f"tb.FRAC_BITS={FRAC_BITS}",
         "-P", f"tb.SAMPLES={SAMPLES}", "-P", f"tb.STRIDES={packed_strides}",
         "-P", f"tb.K={KERNEL_SIZE}", "-P", f"tb.L={LAYERS}", "-P", f"tb.D={DILATION_BASE}", "-P", f"tb.HC={HIDDEN_CHANNELS}",
         "-o", "tb.vvp", str(SV_DIR / "tb.sv")],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    subprocess.run(["vvp", "tb.vvp"], cwd=tmp_path, capture_output=True, text=True, check=True)
    sv_out = read_mem_codes(tmp_path / "encoder_output.mem")

    decimation = math.prod(strides)
    assert WARMUP_ZEROS % decimation == 0, "warmup zeros must keep the stride phase aligned"
    x = torch.tensor([0] * WARMUP_ZEROS + codes_in, dtype=torch.float32) / 2 ** FRAC_BITS
    with torch.no_grad():
        y = model(x.unsqueeze(0))[0, WARMUP_ZEROS // decimation:]
    py_out = torch.floor(y * 2 ** FRAC_BITS + 0.5).int().tolist()

    assert len(sv_out) == len(py_out) == SAMPLES // decimation
    for i, (py_code, sv_code) in enumerate(zip(py_out, sv_out)):
        assert py_code == sv_code, f"Python code {py_code} does not equal SV code {sv_code} at index {i}"
