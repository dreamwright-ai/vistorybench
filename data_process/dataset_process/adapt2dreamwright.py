import json
import os
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vistorybench" / "dataset_loader"))
from dataset_load import StoryDataset

# importing packages from dreamwright for image generation
sys.path.append(str("dreamwright-v2/packages/generators/src"))
sys.path.append(str("dreamwright-v2/packages/core-schemas/src"))
sys.path.append(str("dreamwright-v2/packages/gemini-client/src"))

from dreamwright_generators.character import CharacterGenerator
from dreamwright_generators.location import LocationGenerator
from dreamwright_generators.image import ImageGenerator
from dreamwright_core_schemas import (
    Project, Chapter, Scene, Panel,
    PanelComposition, PanelCharacter, ShotType, CameraAngle,
    Character, CharacterAssets, CharacterDescription,
    Location, LocationAssets, Story
)
from dreamwright_gemini_client import get_extension_for_mime_type

google_api_key = os.environ.get("GOOGLE_API_KEY")
char_generator = CharacterGenerator(api_key=google_api_key)
loc_generator = LocationGenerator(api_key=google_api_key)
img_generator = ImageGenerator(api_key=google_api_key)

def map_genre(vistory_type: str) -> str:
    """
    Map ViStory story type to DreamWright genre enum.

    Valid DreamWright genres: 'romance', 'action', 'fantasy', 'thriller',
    'slice_of_life', 'horror', 'comedy', 'drama', 'mystery', 'scifi'
    """
    type_lower = vistory_type.lower() if vistory_type else ""

    # Direct mappings
    if "romance" in type_lower:
        return "romance"
    elif "action" in type_lower:
        return "action"
    elif "fantasy" in type_lower or "magic" in type_lower:
        return "fantasy"
    elif "thriller" in type_lower or "suspense" in type_lower:
        return "thriller"
    elif "slice" in type_lower or "life" in type_lower or "daily" in type_lower:
        return "slice_of_life"
    elif "horror" in type_lower or "scary" in type_lower:
        return "horror"
    elif "comedy" in type_lower or "funny" in type_lower or "humor" in type_lower:
        return "comedy"
    elif "mystery" in type_lower or "detective" in type_lower:
        return "mystery"
    elif "sci" in type_lower or "fiction" in type_lower or "space" in type_lower:
        return "scifi"
    elif "children" in type_lower or "picture" in type_lower or "kid" in type_lower:
        return "slice_of_life"  # Map children's books to slice_of_life
    else:
        return "drama"  # Default fallback



def slugify(text: str) -> str:
    """Convert text to a valid ID slug."""
    # Lowercase and replace spaces/special chars with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower().strip())
    return slug.strip('-')


def parse_camera_design(camera_text: str) -> dict:
    """
    Parse ViStory camera design text into DreamWright composition fields.

    ViStory camera examples:
    - "Medium shot, eye level"
    - "Close-up, low angle"
    - "Wide shot, bird's eye view"

    Returns:
        dict with shot_type and angle
    """
    camera_lower = camera_text.lower()

    # Determine shot type (order matters - check more specific patterns first)
    shot_type = "medium"  # default
    if "extreme close" in camera_lower:
        shot_type = "extreme_close_up"
    elif "close" in camera_lower or "closeup" in camera_lower:
        shot_type = "close_up"
    elif "wide" in camera_lower or "establishing" in camera_lower:
        shot_type = "wide"
    elif "medium" in camera_lower:
        shot_type = "medium"
    elif "full" in camera_lower:
        shot_type = "full"

    # Determine angle
    angle = "eye_level"  # default
    if "low" in camera_lower:
        angle = "low"
    elif "high" in camera_lower or "bird" in camera_lower:
        angle = "high"
    elif "dutch" in camera_lower or "tilt" in camera_lower:
        angle = "dutch"
    elif "over" in camera_lower and "shoulder" in camera_lower:
        angle = "over_shoulder"

    return {
        "shot_type": shot_type,
        "angle": angle,
        "focus": ""
    }


async def convert_story_to_project(story_id: str, story_data: dict, continuity: bool = True, project_dir: str = "") -> dict:
    """
    Convert a single ViStory story to DreamWright project.json format.

    Args:
        story_id: The story identifier (e.g., "01", "02")
        story_data: Story data from StoryDataset.load_story()
        continuity: Whether to enable panel continuity
        project_dir: The directory where the project and its assets will be saved.

    Returns:
        DreamWright project.json structure as dict
    """
    project_abs_path = Path(project_dir).resolve()
    assets_abs_path = project_abs_path / "assets"

    # --- 1. Generate Character Assets ---
    character_models = {}
    char_id_map = {}
    for char_key, char_info in story_data["characters"].items():
        char_id = f"char_{slugify(char_key)}"
        char_id_map[char_key] = char_id
        
        character_model = Character(
            id=char_id,
            name=char_info.get("name", char_key),
            description=CharacterDescription(physical=char_info.get("prompt", "")),
            assets=CharacterAssets()
        )
        
        reference_image_path = None
        if char_info.get("images") and len(char_info["images"]) > 0:
            img_path = char_info["images"][0]
            if os.path.exists(img_path) and os.path.isfile(img_path):
                reference_image_path = img_path
        
        if reference_image_path:
            print(f"  - Generating assets for character: {char_key}...")
            try:
                sheet_data, sheet_info = await char_generator.generate_character_sheet(
                    character=character_model, style=story_data.get("type", ""), reference_image=Path(reference_image_path))
                
                char_asset_dir = assets_abs_path / "characters" / char_id
                char_asset_dir.mkdir(parents=True, exist_ok=True)
                
                sheet_ext = get_extension_for_mime_type(sheet_info.get("response", {}).get("output", {}).get("mime_type"))
                sheet_filename = f"sheet_default{sheet_ext}"
                sheet_abs_path = char_asset_dir / sheet_filename
                sheet_abs_path.write_bytes(sheet_data)
                character_model.assets.bible_sheet = f"assets/characters/{char_id}/{sheet_filename}"
                character_model.assets.sheet = character_model.assets.bible_sheet

                portrait_data, portrait_info = await char_generator.generate_portrait(
                    character=character_model, style=story_data.get("type", ""), reference_image=sheet_abs_path)
                
                portrait_ext = get_extension_for_mime_type(portrait_info.get("response", {}).get("output", {}).get("mime_type"))
                portrait_filename = f"portrait{portrait_ext}"
                (char_asset_dir / portrait_filename).write_bytes(portrait_data)
                character_model.assets.portrait = f"assets/characters/{char_id}/{portrait_filename}"
            except Exception as e:
                print(f"    ...Error generating assets for character {char_key}: {e}")
        
        character_models[char_id] = character_model

    # --- 2. Generate Location Assets ---
    location_models = {}
    location_id_map = {}
    unique_scene_descs = {}
    for shot in story_data["shots"]:
        scene_desc = shot.get("scene", "")
        if scene_desc:
            scene_hash = slugify(scene_desc[:50])
            if scene_hash not in location_id_map:
                loc_id = f"loc_{scene_hash}"
                location_id_map[scene_hash] = loc_id
                unique_scene_descs[loc_id] = scene_desc
    
    for loc_id, scene_desc in unique_scene_descs.items():
        location_model = Location(id=loc_id, name=scene_desc[:100], description=scene_desc)
        print(f"  - Generating assets for location: {scene_desc[:50]}...")
        try:
            image_data, image_info = await loc_generator.generate_reference(
                location=location_model, style=story_data.get("type", ""))
            
            loc_asset_dir = assets_abs_path / "locations" / loc_id
            loc_asset_dir.mkdir(parents=True, exist_ok=True)
            
            image_ext = get_extension_for_mime_type(image_info.get("response", {}).get("output", {}).get("mime_type"))
            image_filename = f"reference{image_ext}"
            (loc_asset_dir / image_filename).write_bytes(image_data)
            location_model.assets = LocationAssets(reference=f"assets/locations/{loc_id}/{image_filename}")
        except Exception as e:
            print(f"    ...Error generating assets for location {loc_id}: {e}")
            location_model.assets = LocationAssets()
        
        location_models[loc_id] = location_model

    # --- 3. Build Panel, Scene, and Chapter Models ---
    panel_models = []
    for shot_idx, shot in enumerate(story_data["shots"]):
        comp_dict = parse_camera_design(shot.get("camera", ""))
        composition = PanelComposition(
            shot_type=ShotType(comp_dict["shot_type"]),
            angle=CameraAngle(comp_dict["angle"])
        )

        panel_characters = [
            PanelCharacter(character_id=char_id_map[key], expression="neutral", pose="", position="center")
            for key in shot.get("character_key", []) if key in char_id_map
        ]
        
        action = " ".join(part for part in [shot.get("plot"), shot.get("script")] if part)
        
        panel_models.append(Panel(
            id=f"ch1_s1_p{shot_idx + 1}",
            number=shot_idx + 1,
            composition=composition,
            characters=panel_characters,
            action=action,
            continues_from_previous=continuity and shot_idx > 0
        ))
        
    first_scene_desc = story_data["shots"][0].get("scene", "") if story_data["shots"] else ""
    first_scene_hash = slugify(first_scene_desc[:50]) if first_scene_desc else ""
    scene_location_id = location_id_map.get(first_scene_hash)
    if not scene_location_id and location_models:
        scene_location_id = list(location_models.keys())[0]

    scene = Scene(
        id="ch1_s1", number=1, location_id=scene_location_id,
        description=first_scene_desc, panels=panel_models,
        character_ids=list(character_models.keys())
    )
    chapter = Chapter(id="ch1", number=1, title=f"ViStory {story_id}", scenes=[scene])

    # --- 4. Assemble Full Project Model ---
    project = Project(
        id=f"vistory_{story_id}",
        name=f"ViStory Story {story_id}",
        style=story_data.get("type", ""),
        characters=list(character_models.values()),
        locations=list(location_models.values()),
        chapters=[chapter],
        story=Story(
            id=f"story_vistory_{story_id}",
            title=f"ViStory Story {story_id}",
            genre=map_genre(story_data.get("type", ""))
        )
    )

    # --- 5. Generate Panel Images ---
    print(f"  - Generating panel images for story {story_id}...")
    char_refs = {c.id: assets_abs_path.parent / c.assets.bible_sheet for c in project.characters if c.assets.bible_sheet}
    loc_refs = {l.id: assets_abs_path.parent / l.assets.reference for l in project.locations if l.assets.reference}
    
    try:
        await img_generator.generate_scene_panels(
            scene=project.chapters[0].scenes[0],
            chapter_number=project.chapters[0].number,
            characters=character_models,
            locations=location_models,
            character_references=char_refs,
            location_references=loc_refs,
            output_dir=assets_abs_path,
            style=project.style,
            overwrite=True
        )
        print(f"    ...Panel images generated for story {story_id}")
    except Exception as e:
        print(f"    ...Error generating panel images for story {story_id}: {e}")

    # --- 6. Return final project structure as dict ---
    return project.model_dump(mode='json')


class DreamWrightAdapter:
    """Adapter class to convert ViStory dataset to DreamWright projects."""

    def __init__(self, dataset_path: str, output_path: str, dreamwright_path: str):
        """
        Initialize the adapter.

        Args:
            dataset_path: Path to ViStory dataset
            output_path: Path to output DreamWright projects (typically dreamwright/projects/)
            dreamwright_path: Path to DreamWright repository
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dreamwright_path = dreamwright_path
        self.dataset = StoryDataset(dataset_path)

    async def convert(self, story_list: list, language: str, continuity: bool = True) -> dict:
        """
        Convert multiple stories to DreamWright projects.

        Args:
            story_list: List of story IDs to convert
            language: "en" or "ch"
            continuity: Whether to enable panel continuity

        Returns:
            Manifest dict with converted project info
        """
        manifest = {
            "created_at": datetime.now().isoformat(),
            "language": language,
            "source": "ViStory",
            "projects": {}
        }

        try:
            stories_data = self.dataset.load_stories(story_list, language)
        except Exception as e:
            print(f"Error loading stories: {e}")
            return manifest
        
        conversion_tasks = []
        for story_id in story_list:
            if story_id not in stories_data:
                print(f"Warning: Story {story_id} not found, skipping")
                continue

            story_data = stories_data[story_id]
            print(f"Queuing story {story_id} for conversion...")

            project_id = f"vistory_{story_id}"
            project_dir = os.path.join(self.output_path, project_id)

            task = asyncio.create_task(
                convert_story_to_project(
                    story_id, story_data, continuity, project_dir
                )
            )
            conversion_tasks.append(task)
        
        print(f"\nRunning {len(conversion_tasks)} conversions in parallel...")
        converted_projects = await asyncio.gather(*conversion_tasks)
        print("...All conversions complete.\n")

        for project in converted_projects:
            story_id = project["id"].replace("vistory_", "")
            project_dir = os.path.join(self.output_path, project["id"])

            try:
                os.makedirs(project_dir, exist_ok=True)
                project_json_path = os.path.join(project_dir, "project.json")
                with open(project_json_path, 'w', encoding='utf-8') as f:
                    json.dump(project, f, indent=2, ensure_ascii=False)
                print(f"  Saved: {project_json_path}")
            except (OSError, IOError) as e:
                print(f"  Error saving project {story_id}: {e}")
                continue

            # Add to manifest
            manifest["projects"][story_id] = {
                "project_id": project["id"],
                "project_path": project_dir,
                "num_characters": len(project["characters"]),
                "num_locations": len(project["locations"]),
                "num_panels": len(project["chapters"][0]["scenes"][0]["panels"])
            }

        return manifest


async def main():
    parser = argparse.ArgumentParser(
        description='Convert ViStory dataset to DreamWright project format'
    )
    parser.add_argument('--language', type=str, choices=['en', 'ch'],
                        default='en', help='Language: en (English) or ch (Chinese)')
    parser.add_argument('--stories', nargs='+', type=str, default=None,
                        help='Specific story IDs to convert (default: all)')
    parser.add_argument('--split', type=str, choices=['full', 'lite'], default='full',
                        help='Dataset split to use')
    parser.add_argument('--continuity', action='store_true', default=True,
                        help='Enable panel continuity (default: True)')
    parser.add_argument('--no-continuity', action='store_false', dest='continuity',
                        help='Disable panel continuity')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--dreamwright_path', type=str,
                        default=os.environ.get('DREAMWRIGHT_PATH', 'dreamwright-v2'),
                        help='Path to DreamWright repository (or set DREAMWRIGHT_PATH env var)')

    args = parser.parse_args()

    # Determine paths
    if args.data_path:
        data_path = args.data_path
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).resolve().parent
        data_path = script_dir.parent.parent / "data"

    dataset_path = os.path.join(data_path, "dataset", "ViStory")
    output_path = os.path.join(args.dreamwright_path, "projects")

    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Language: {args.language}")
    print(f"Continuity: {args.continuity}")

    # Check paths exist
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if not os.path.exists(args.dreamwright_path):
        print(f"Error: DreamWright path does not exist: {args.dreamwright_path}")
        sys.exit(1)

    # Initialize adapter
    adapter = DreamWrightAdapter(dataset_path, output_path, args.dreamwright_path)

    # Get story list
    if args.stories:
        story_list = args.stories
    else:
        story_list = adapter.dataset.get_story_name_list(split=args.split)

    print(f"Found {len(story_list)} stories: {story_list[:5]}{'...' if len(story_list) > 5 else ''}")

    # Convert stories
    manifest = await adapter.convert(story_list, args.language, args.continuity)

    # Save manifest
    manifest_path = os.path.join(
        data_path, "dataset_processed", "dreamwright",
        f"ViStory_{args.language}", "manifest.json"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest saved: {manifest_path}")
    print(f"Converted {len(manifest['projects'])} stories successfully")


if __name__ == "__main__":
    asyncio.run(main())
