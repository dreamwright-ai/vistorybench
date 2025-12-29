"""
DreamWright inference script for ViStoryBench.

This script runs DreamWright's image generation on converted ViStory projects
and collects outputs in the benchmark's expected format.

Usage:
    python inference_custom.py --language en
    python inference_custom.py --language en --stories 01 02 --skip-assets
"""

import json
import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


# DreamWright repository path (can be overridden via env var or CLI arg)
# Default to the submodule in the current directory if not set
DEFAULT_DREAMWRIGHT_PATH = os.environ.get('DREAMWRIGHT_PATH', os.path.abspath('dreamwright-v2'))
DREAMWRIGHT_PATH = DEFAULT_DREAMWRIGHT_PATH

# Required environment variables for DreamWright image generation
REQUIRED_ENV_VARS = ['GOOGLE_API_KEY']


def check_environment() -> Tuple[bool, list]:
    """
    Check that required environment variables are set.

    Returns:
        Tuple of (all_present, missing_vars)
    """
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    return len(missing) == 0, missing


def run_dreamwright_command(cmd: list, cwd: str = None,
                             timeout: int = 300) -> Tuple[bool, str]:
    """
    Run a DreamWright CLI command.

    Args:
        cmd: Command list (without 'uv run')
        cwd: Working directory (defaults to global DREAMWRIGHT_PATH)
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, output/error message)
    """
    if cwd is None:
        cwd = DREAMWRIGHT_PATH

    full_cmd = ["uv", "run"] + cmd

    try:
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            # Check if this is an "asset already exists" error - treat as success
            combined_output = result.stdout + result.stderr
            if "already exists" in combined_output.lower():
                return True, "Asset already exists (skipped)"
            return False, f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def generate_character(project_id: str, char_name: str,
                       reference_image: str = None, style: str = None) -> bool:
    """Generate character sheet for a project."""
    cmd = ["dreamwright", "generate", "character",
           "--name", char_name, "-p", project_id]

    if style:
        cmd.extend(["--style", style])

    if reference_image and os.path.exists(reference_image):
        cmd.extend(["--reference", reference_image])

    print(f"    Generating character: {char_name}")
    success, msg = run_dreamwright_command(cmd)

    if success and "already exists" in msg.lower():
        print(f"      (already exists, skipping)")
    elif not success:
        print(f"      Warning: {msg[:200]}")  # Truncate long error messages

    return success


def generate_location(project_id: str, location_name: str, style: str = None) -> bool:
    """Generate location background for a project."""
    cmd = ["dreamwright", "generate", "location",
           "--name", location_name, "-p", project_id]

    if style:
        cmd.extend(["--style", style])

    print(f"    Generating location: {location_name[:50]}...")
    success, msg = run_dreamwright_command(cmd)

    if success and "already exists" in msg.lower():
        print(f"      (already exists, skipping)")
    elif not success:
        print(f"      Warning: {msg[:200]}")  # Truncate long error messages

    return success


def generate_panel(project_id: str, panel_id: str, overwrite: bool = False, style: str = None) -> bool:
    """Generate a panel image for a project."""
    cmd = ["dreamwright", "generate", "image", panel_id, "-p", project_id]

    if style:
        cmd.extend(["--style", style])

    if overwrite:
        cmd.append("--overwrite")

    print(f"    Generating panel: {panel_id}")
    success, msg = run_dreamwright_command(cmd, timeout=120)

    if success and "already exists" in msg.lower():
        print(f"      (already exists, skipping)")
    elif not success:
        print(f"      Warning: {msg[:200]}")  # Truncate long error messages

    return success


def run_inference(manifest: dict, output_base_path: str, language: str,
                  skip_assets: bool = False, skip_existing: bool = True) -> dict:
    """
    Run inference for all projects in the manifest.

    Args:
        manifest: Manifest from adapt2dreamwright.py
        output_base_path: Base path for benchmark outputs
        language: Language code
        skip_assets: Skip character/location generation
        skip_existing: Skip panels that already have images

    Returns:
        Results dict with generation stats
    """
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "language": language,
        "stories": {}
    }

    timestamp = results["timestamp"]

    for story_id, project_info in manifest["projects"].items():
        project_id = project_info["project_id"]
        project_path = project_info["project_path"]

        print(f"\nProcessing story {story_id} (project: {project_id})")

        # Load project.json to get details
        project_json_path = os.path.join(project_path, "project.json")
        if not os.path.exists(project_json_path):
            print(f"  Warning: project.json not found at {project_json_path}")
            continue

        try:
            with open(project_json_path, 'r', encoding='utf-8') as f:
                project = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error loading project.json: {e}")
            continue

        # Get style from project
        style = project.get("style", "webtoon")
        print(f"  Using style: {style}")

        story_results = {
            "project_id": project_id,
            "characters_generated": 0,
            "locations_generated": 0,
            "locations_failed": 0,
            "panels_generated": 0,
            "panels_failed": 0
        }

        num_locations = len(project.get("locations", []))

        # Step 1: Generate character sheets (if not skipping)
        if not skip_assets:
            print("  Step 1: Generating character sheets...")
            for char in project.get("characters", []):
                ref_image = char.get("assets", {}).get("reference_input")
                if generate_character(project_id, char["name"], ref_image, style=style):
                    story_results["characters_generated"] += 1

        # Step 2: Generate location backgrounds (if not skipping)
        if not skip_assets:
            print("  Step 2: Generating location backgrounds...")
            for loc in project.get("locations", []):
                if generate_location(project_id, loc["name"], style=style):
                    story_results["locations_generated"] += 1
                else:
                    story_results["locations_failed"] += 1

            # Check if location generation failed - this will cause panel generation to fail
            if story_results["locations_failed"] > 0:
                print(f"  ERROR: {story_results['locations_failed']}/{num_locations} locations failed to generate")
                print("  Panel generation requires location assets. Skipping panels for this story.")
                print("  Tip: Ensure GOOGLE_API_KEY is set correctly.")
                results["stories"][story_id] = story_results
                continue

        # Step 3: Generate panel images
        print("  Step 3: Generating panel images...")

        # Create output directory for this story
        output_dir = os.path.join(
            output_base_path, "dreamwright", "base", language,
            timestamp, story_id
        )
        os.makedirs(output_dir, exist_ok=True)

        # Get panels from project
        panels = []
        for chapter in project.get("chapters", []):
            for scene in chapter.get("scenes", []):
                panels.extend(scene.get("panels", []))

        for panel_idx, panel in enumerate(panels):
            panel_id = panel["id"]

            # Define where we expect the image to be (or where it might be)
            # DreamWright naming conventions can vary slightly
            possible_paths = [
                os.path.join(DREAMWRIGHT_PATH, "projects", project_id,
                            panel.get("image_path") or f"assets/panels/ch1/s1_p{panel_idx + 1}.jpg"),
                os.path.join(DREAMWRIGHT_PATH, "projects", project_id,
                            "assets", "panels", "ch1", f"s1_p{panel_idx + 1}.jpg"),
                os.path.join(DREAMWRIGHT_PATH, "projects", project_id,
                            "assets", "panels", "chapter-1", "scene-1", f"panel-{panel_idx + 1}.jpg"),
            ]

            # Find existing image
            src_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    src_path = path
                    break

            if skip_existing and src_path:
                print(f"    Panel {panel_id} already exists at {os.path.basename(src_path)}, skipping generation")
            else:
                # Generate the panel
                if not generate_panel(project_id, panel_id, style=style):
                    story_results["panels_failed"] += 1
                    continue
                
                # Re-check paths after generation
                src_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        src_path = path
                        break

            if src_path:
                dst_path = os.path.join(output_dir, f"{panel_idx:02d}.jpg")
                try:
                    shutil.copy(src_path, dst_path)
                    story_results["panels_generated"] += 1
                    print(f"    Copied: {dst_path}")
                except (OSError, IOError) as e:
                    print(f"    Error copying {panel_id}: {e}")
                    story_results["panels_failed"] += 1
            else:
                print(f"    Warning: Could not find generated image for {panel_id}")
                story_results["panels_failed"] += 1

        results["stories"][story_id] = story_results
        print(f"  Completed: {story_results['panels_generated']}/{len(panels)} panels")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run DreamWright inference for ViStoryBench'
    )
    parser.add_argument('--language', type=str, choices=['en', 'ch'],
                        default='en', help='Language: en (English) or ch (Chinese)')
    parser.add_argument('--stories', nargs='+', type=str, default=None,
                        help='Specific story IDs to process (default: all in manifest)')
    parser.add_argument('--skip-assets', action='store_true',
                        help='Skip character/location generation (assume already done)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip panels that already have images')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate all panels even if they exist')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--dreamwright_path', type=str,
                        default=DEFAULT_DREAMWRIGHT_PATH,
                        help='Path to DreamWright repository (or set DREAMWRIGHT_PATH env var)')

    args = parser.parse_args()

    global DREAMWRIGHT_PATH
    DREAMWRIGHT_PATH = args.dreamwright_path

    # Check required environment variables
    env_ok, missing_vars = check_environment()
    if not env_ok:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nTo fix:")
        print("  export GOOGLE_API_KEY='your-google-api-key'")
        print("  # Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    # Determine paths
    if args.data_path:
        data_path = args.data_path
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).resolve().parent
        data_path = script_dir.parent.parent.parent / "dreamwright-v2" / "projects"

    # Load manifest
    manifest_path = os.path.join(
        data_path, "dataset_processed", "dreamwright",
        f"ViStory_{args.language}", "manifest.json"
    )

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run adapt2dreamwright.py first to convert the dataset")
        sys.exit(1)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"Loaded manifest with {len(manifest['projects'])} projects")
    print(f"DreamWright path: {DREAMWRIGHT_PATH}")

    # Filter stories if specified
    if args.stories:
        filtered_projects = {
            sid: info for sid, info in manifest["projects"].items()
            if sid in args.stories
        }
        manifest["projects"] = filtered_projects
        print(f"Filtered to {len(manifest['projects'])} stories: {args.stories}")

    # Check DreamWright path
    if not os.path.exists(DREAMWRIGHT_PATH):
        print(f"Error: DreamWright path does not exist: {DREAMWRIGHT_PATH}")
        sys.exit(1)

    # Run inference
    output_base_path = os.path.join(data_path, "outputs")
    skip_existing = not args.regenerate and args.skip_existing

    results = run_inference(
        manifest,
        output_base_path,
        args.language,
        skip_assets=args.skip_assets,
        skip_existing=skip_existing
    )

    # Save results
    results_path = os.path.join(
        output_base_path, "dreamwright", "base", args.language,
        results["timestamp"], "inference_results.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Inference complete!")
    print(f"Results saved: {results_path}")
    print(f"Output directory: {os.path.dirname(results_path)}")

    # Print summary
    total_generated = sum(s["panels_generated"] for s in results["stories"].values())
    total_failed = sum(s["panels_failed"] for s in results["stories"].values())
    total_loc_failed = sum(s.get("locations_failed", 0) for s in results["stories"].values())
    print(f"\nSummary:")
    print(f"  Stories processed: {len(results['stories'])}")
    print(f"  Panels generated: {total_generated}")
    print(f"  Panels failed: {total_failed}")
    print(f"  Locations failed: {total_loc_failed}")

    if total_loc_failed > 0:
        print(f"\nNote: Location generation failed for some stories.")
        print("  Ensure GOOGLE_API_KEY is set: export GOOGLE_API_KEY='your-key'")
        print("  Get an API key at: https://aistudio.google.com/apikey")

    if total_failed > 0:
        print(f"\nNote: Some panels failed to generate. Check the logs above.")

    print(f"\nNext step: Run evaluation with:")
    print(f"  python bench_run.py --method dreamwright --language {args.language} --timestamp {results['timestamp']}")


if __name__ == "__main__":
    main()
