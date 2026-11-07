# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import time

from tpu_sync.rpc.raiden_controller import RaidenController, RaidenControllerServer


def main():
    parser = argparse.ArgumentParser(description="Start Raiden Controller Server")
    parser.add_argument(
        "--port", type=int, required=True, help="Port to bind the controller server"
    )
    args = parser.parse_args()

    print(f"Starting RaidenController on port {args.port}...", flush=True)
    controller = RaidenController(port=args.port)
    server = RaidenControllerServer(controller)
    server.start()
    print(f"RaidenControllerServer is running on port {args.port}.", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping RaidenControllerServer...", flush=True)
        server.stop()


if __name__ == "__main__":
    main()
