import argparse
import time

from tpu_raiden.rpc.raiden_controller import (RaidenController,
                                              RaidenControllerServer)


def main():
    parser = argparse.ArgumentParser(
        description="Start Raiden Controller Server")
    parser.add_argument("--port",
                        type=int,
                        required=True,
                        help="Port to bind the controller server")
    args = parser.parse_args()

    print(f"Starting RaidenController on port {args.port}...", flush=True)
    controller = RaidenController(port=args.port)
    server = RaidenControllerServer(controller)
    server.start()
    print(f"RaidenControllerServer is running on port {args.port}.",
          flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping RaidenControllerServer...", flush=True)
        server.stop()


if __name__ == "__main__":
    main()
