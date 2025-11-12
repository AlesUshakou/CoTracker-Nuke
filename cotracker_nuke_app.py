#!/usr/bin/env python3
"""
CoTracker Nuke Integration App - Refactored
===========================================

This is the new modular version of the CoTracker Nuke integration app.
The original monolithic file has been refactored into a clean modular structure.

Features:
- Modular architecture with separate concerns
- Proper logging with configurable verbosity levels
- Clean separation of UI, core logic, and exporters
- Support for both GUI and CLI interfaces

Usage:
    python cotracker_nuke_app_new.py [--log-level DEBUG|INFO|WARNING|ERROR]

Author: AI Assistant (under human supervision)
License: MIT
"""

#!/usr/bin/env python3
"""
CoTracker Nuke Integration App - Refactored (with --video prefill support)
"""

import os
import multiprocessing as mp
import argparse
import sys
import inspect
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cotracker_nuke import create_gradio_interface


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="CoTracker Nuke Integration - Modular Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level (default: INFO)"
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug file logging"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="CoTracker Nuke Integration v1.0.0 (Modular)"
    )

    parser.add_argument(
        "--video", "-V",
        help="(optional) path to a video file to pre-fill the Upload field in the UI",
        default=None
    )

    args = parser.parse_args()

    try:
        print("CoTracker Nuke Integration - Modular Version")
        print(f"Console log level: {args.log_level}")
        print(f"Debug logging: {'Disabled' if args.no_debug else 'Enabled'}")
        if args.video:
            print(f"Video prefill requested: {args.video}")
        print()

        # If a video path was provided, expose it to the UI in multiple ways:
        # 1) set an environment variable that UI code can read
        # 2) attempt to pass it as a keyword argument to create_gradio_interface
        if args.video:
            # 1) env var
            os.environ.setdefault('COTRACKER_VIDEO', args.video)

        debug_mode = not args.no_debug

        # 2) try to detect create_gradio_interface signature and pass video if supported
        kwargs = {}
        try:
            sig = inspect.signature(create_gradio_interface)
            params = sig.parameters
            if args.video:
                if 'prefill_video' in params:
                    kwargs['prefill_video'] = args.video
                elif 'video' in params:
                    kwargs['video'] = args.video
                elif 'video_path' in params:
                    kwargs['video_path'] = args.video
        except Exception:
            # if signature inspection fails, continue without kwargs
            kwargs = {}

        # Create and launch Gradio interface
        if kwargs:
            interface = create_gradio_interface(debug_mode, args.log_level, **kwargs)
        else:
            # fallback to the normal call
            interface = create_gradio_interface(debug_mode, args.log_level)

        # Launch the interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True
        )

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"Error starting application: {str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
