UNQUANTIZED = "unquantized"
MXFP4 = "mxfp4"
AWQ = "awq"
FP8 = "fp8"
COMPRESSED_TENSORS = "compressed-tensors"
# vLLM's native detection name for ModelOpt NVFP4 checkpoints
# (ModelOptNvFp4Config.get_name()).
NVFP4 = "modelopt_fp4"


def get_tpu_quant_method(quant_method: str) -> str:
    return "tpu-" + quant_method
