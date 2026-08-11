import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time

import requests

# Configure basic logging to ensure we get clear output in CI
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _run_online_profiler_test(is_phased: bool):
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(
            f"Using temporary directory for profiles (phased={is_phased}): {temp_dir}"
        )

        # 1. Start server as a subprocess with a lightweight model
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server", "--model",
            "facebook/opt-125m", "--port", "8000",
            "--profiler-config.profiler", "torch",
            "--profiler-config.torch_profiler_dir", temp_dir,
            "--profiler-config.ignore_frontend", "true", "--max-model-len",
            "128", "--gpu-memory-utilization", "0.2"
        ]

        # Add delay_iterations so phased profiler runs predictably quickly in the test
        if is_phased:
            cmd.extend(["--profiler-config.delay_iterations", "1"])

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        env = os.environ.copy()
        if is_phased:
            env["USE_PHASED_PROFILER"] = "1"

        # We purposely do not capture stdout/stderr so it streams naturally into the CI logs
        proc = subprocess.Popen(cmd, env=env)

        server_url = "http://localhost:8000"

        try:
            # Wait for server to pass healthcheck (timeout 360s)
            logger.info("Waiting for server to become healthy...")
            server_ready = False
            for step in range(360):
                try:
                    res = requests.get(f"{server_url}/health", timeout=2)
                    if res.status_code == 200:
                        server_ready = True
                        break
                except requests.RequestException:
                    pass

                if step > 0 and step % 10 == 0:
                    logger.info(
                        f"Still waiting for server... ({step}s elapsed)")
                time.sleep(1)

            if not server_ready:
                raise RuntimeError(
                    "Server failed to start and pass health check in expected time (360s). Check CI logs above for server crash details."
                )

            # 2. Arm the profiler with advanced parameters
            profile_opts = (
                "host_tracer_level:2;"
                "device_tracer_level:1;"
                "tpu_trace_mode:TRACE_COMPUTE;"
                "tpu_perf_counters:true;"
                "tpu_enable_periodic_counter_sampling:true;"
                "e2e_enable_fw_throttle_event:true;"
                "e2e_enable_fw_power_level_event:true;"
                "e2e_enable_fw_thermal_event:true;"
                "tpu_tc_perf_counter_sampling_options:interval_us:10000 scaling:0 counter_size_bits:3 indices:0 indices:1 indices:3 indices:6 indices:8 indices:10 indices:12 indices:14 indices:18 indices:31 indices:40 indices:50 indices:56 indices:57 indices:73 indices:203;"
            )
            logger.info(
                "Server is healthy. Arming the profiler via /start_profile JSON payload"
            )
            res = requests.post(f"{server_url}/start_profile",
                                json={"profile_prefix": profile_opts},
                                timeout=180)
            assert res.status_code == 200, f"/start_profile failed with status {res.status_code}: {res.text}"
            logger.info(
                "Profiler armed successfully with advanced prefix options.")

            # 3. Fire an inference request to capture trace data
            logger.info("Sending inference request to generate traffic...")
            gen_req = {
                "model": "facebook/opt-125m",
                "prompt": "The future of AI is",
                "max_tokens": 10
            }
            # Send enough requests to trigger phases (specifically prefill vs decode)
            for _ in range(6):
                res = requests.post(f"{server_url}/v1/completions",
                                    json=gen_req,
                                    timeout=30)
                assert res.status_code == 200, f"Inference failed with status {res.status_code}: {res.text}"
            logger.info("Inference completed successfully.")

            # 4. Stop the profiler
            logger.info("Stopping the profiler via /stop_profile...")
            res = requests.post(f"{server_url}/stop_profile", timeout=180)
            assert res.status_code == 200, f"/stop_profile failed with status {res.status_code}: {res.text}"
            logger.info("Profiler stopped successfully.")

            # 5. Verify the profile was generated
            logger.info("Verifying generated traces...")
            pb_files = glob.glob(f"{temp_dir}/**/*.pb", recursive=True)
            assert len(
                pb_files
            ) > 0, f"No xplane.pb files found in the trace directory {temp_dir}!"
            logger.info(f"Found {len(pb_files)} trace artifact(s): {pb_files}")

            # 6. Copy the trace to an artifact directory for download from CI
            prefix = "online_phased" if is_phased else "online_non_phased"
            artifact_dir = "/tmp/vllm_profile_artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            for f in pb_files:
                rel_path = os.path.relpath(f, temp_dir)
                flattened_name = rel_path.replace(os.sep, "_")
                dest = os.path.join(artifact_dir, f"{prefix}_{flattened_name}")
                shutil.copy(f, dest)
                os.chmod(dest, 0o666)
                logger.info(f"Copied trace artifact to: {dest}")

        finally:
            logger.info("Tearing down the server subprocess...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server did not terminate gracefully, sending SIGKILL...")
                proc.kill()
                proc.wait(timeout=5)
            logger.info("Server shutdown complete.")

            lockfile_path = "/tmp/libtpu_lockfile"
            if os.path.exists(lockfile_path):
                try:
                    os.remove(lockfile_path)
                    logger.info(f"Removed stale lockfile: {lockfile_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove lockfile: {e}")


def _run_offline_profiler_test(is_phased: bool):
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(
            f"Using temporary directory for OFFLINE profiles (phased={is_phased}): {temp_dir}"
        )
        script_path = os.path.join(temp_dir, "offline_profiling_script.py")

        script_content = f"""
import os
import shutil
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
lockfile_path = "/tmp/libtpu_lockfile"
if os.path.exists(lockfile_path):
    try:
        os.remove(lockfile_path)
    except OSError:
        pass

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

profile_opts = (
    "host_tracer_level:2;"
    "device_tracer_level:1;"
    "tpu_trace_mode:TRACE_COMPUTE;"
    "tpu_perf_counters:true;"
    "tpu_enable_periodic_counter_sampling:true;"
    "e2e_enable_fw_throttle_event:true;"
    "e2e_enable_fw_power_level_event:true;"
    "e2e_enable_fw_thermal_event:true;"
    "tpu_tc_perf_counter_sampling_options:interval_us:10000 scaling:0 counter_size_bits:3 indices:0 indices:1 indices:3 indices:6 indices:8 indices:10 indices:12 indices:14 indices:18 indices:31 indices:40 indices:50 indices:56 indices:57 indices:73 indices:203;"
)

pconfig_kwargs = {{
    "profiler": "torch",
    "torch_profiler_dir": "{temp_dir}",
    "ignore_frontend": True,
}}
if {is_phased}:
    pconfig_kwargs["delay_iterations"] = 1

profiler_config = ProfilerConfig(**pconfig_kwargs)

if __name__ == "__main__":
    print("Initializing offline LLM for profiling test")
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=128,
        gpu_memory_utilization=0.2,
        profiler_config=profiler_config
    )

    print(f"Starting profile with prefix: {{profile_opts}}")
    llm.start_profile(profile_prefix=profile_opts)

    print("Generating traffic...")
    prompts = ["The future of AI is"] * 4
    for _ in range(6):
        llm.generate(prompts, SamplingParams(max_tokens=10))

    print("Stopping profile...")
    llm.stop_profile()
    os._exit(0)
"""
        with open(script_path, "w") as f:
            f.write(script_content)

        env = os.environ.copy()
        if is_phased:
            env["USE_PHASED_PROFILER"] = "1"

        logger.info("Starting isolated OFFLINE vLLM script...")
        try:
            proc = subprocess.run(["python", script_path],
                                  env=env,
                                  timeout=600)
            if proc.returncode != 0:
                logger.error("OFFLINE SCRIPT FAILED!")
                raise RuntimeError(
                    f"Offline script failed with exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.error(
                "OFFLINE SCRIPT TIMED OUT AFTER 600s! VFIO lock may be lingering."
            )
            raise

        logger.info(
            "Sleeping for 15s to allow kernel to gracefully reclaim TPU VFIO locks..."
        )
        import time
        time.sleep(15)

        logger.info(
            "Offline script completed successfully. Verifying generated traces..."
        )
        pb_files = glob.glob(f"{temp_dir}/**/*.pb", recursive=True)
        assert len(
            pb_files
        ) > 0, f"No xplane.pb files found in the trace directory {temp_dir}!"
        logger.info(f"Found {len(pb_files)} trace artifact(s): {pb_files}")

        prefix = "offline_phased" if is_phased else "offline_non_phased"
        artifact_dir = "/tmp/vllm_profile_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        for f in pb_files:
            rel_path = os.path.relpath(f, temp_dir)
            flattened_name = rel_path.replace(os.sep, "_")
            dest = os.path.join(artifact_dir, f"{prefix}_{flattened_name}")
            shutil.copy(f, dest)
            os.chmod(dest, 0o666)
            logger.info(f"Copied trace artifact to: {dest}")
