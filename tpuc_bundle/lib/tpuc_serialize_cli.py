"""Runs `tpuc_serialize_executable` out of process for deepsea_torch_bridge.

argv: asm_runfile entry module device_id out_path num_params
      param_shape... result_shape...
"""

import ctypes
import os
import sys


def main():
  asm, entry, module, device_id, out_path, num_params = sys.argv[1:7]
  shapes = sys.argv[7:]
  if os.environ.get("TPUC_SERIALIZE_ARGS_DUMP"):
    import json  # pylint: disable=g-import-not-at-top
    with open(os.environ["TPUC_SERIALIZE_ARGS_DUMP"], "w") as f:
      json.dump(sys.argv[1:], f)
  n = int(num_params)
  params, results = shapes[:n], shapes[n:]
  lib = ctypes.CDLL(
      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "libtpuc_serializer.so"),
      mode=ctypes.RTLD_LOCAL,
  )
  fn = lib.tpuc_serialize_executable
  fn.restype = ctypes.c_int
  fn.argtypes = [
      ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
      ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
      ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
      ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  p_arr = (ctypes.c_char_p * max(1, len(params)))(*[s.encode() for s in params])
  r_arr = (ctypes.c_char_p * max(1, len(results)))(
      *[s.encode() for s in results])
  payload = ctypes.c_void_p()
  size = ctypes.c_size_t()
  err = ctypes.c_char_p()
  rc = fn(asm.encode(), entry.encode(), module.encode(), len(params), p_arr,
          len(results), r_arr, int(device_id), ctypes.byref(payload),
          ctypes.byref(size), ctypes.byref(err))
  if rc != 0:
    sys.stderr.write(f"tpuc_serialize_executable failed: {err.value!r}\n")
    sys.stderr.flush()
    os._exit(1)
  with open(out_path, "wb") as f:
    f.write(ctypes.string_at(payload, size.value))
  sys.stdout.flush()
  # Skip interpreter / google3 static teardown.
  os._exit(0)


if __name__ == "__main__":
  main()
