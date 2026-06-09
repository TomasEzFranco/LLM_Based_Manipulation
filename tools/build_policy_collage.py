#!/usr/bin/env python3
"""Build a printable photo/command/thought contact sheet from a CSV manifest.

Workflow:
1. Generate an editable manifest from a policy-thought JSON export.
2. Fill in only the photo/frame paths you actually want to show.
3. Build a printable HTML collage. Missing images become visible placeholders.

This script intentionally uses only the Python standard library so it can run
before photos, video frames, or extra packages are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_POLICY_JSON = "results/logs_photos_policy_thoughts_prompts_1_2_3.json"
DEFAULT_IMPORTANT_COMMANDS = (
    "pick_placed_left",
    "pick_placed_right",
    "pick_other",
    "grasp_cube",
    "place_left",
    "place_right",
    "place_left_stack",
    "place_right_stack",
    "return_cube",
    "verify_last_place",
    "stop_run",
)
MANIFEST_FIELDS = [
    "include",
    "prompt",
    "command_index",
    "image_path",
    "command",
    "thought",
    "note",
    "video_path",
    "video_time_s",
    "tile_title",
]


@dataclass(frozen=True)
class PolicyCommand:
    prompt: str
    prompt_id: str
    command_index: int
    command: str
    thought: str
    reason: str
    timestamp_local: str


@dataclass
class Tile:
    prompt: str
    command_index: str
    title: str
    command: str
    thought: str
    note: str
    image_src: str
    image_status: str
    source_image: str
    video_path: str
    video_time_s: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_text: str, *, base_dir: Path | None = None) -> Path:
    path = Path(str(path_text)).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (project_root() / path).resolve()


def normalize_prompt(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = re.sub(r"(?i)^prompt[\s_-]*", "", text).strip()
    if re.fullmatch(r"\d+[A-Za-z]", compact):
        return compact.upper()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text.lower().replace("prompt_", "").replace("prompt ", "").strip()


def coerce_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def trim_text(value: str, max_chars: int) -> str:
    text = clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def truthy_include(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text in {"", "1", "true", "yes", "y", "include"}:
        return True
    return text not in {"0", "false", "no", "n", "skip", "exclude"}


def load_policy_commands(policy_json: Path) -> dict[tuple[str, int], PolicyCommand]:
    if not policy_json.exists():
        raise SystemExit(f"ERROR: policy JSON not found: {policy_json}")
    try:
        payload = json.loads(policy_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid policy JSON in {policy_json}: {exc}") from exc
    commands: dict[tuple[str, int], PolicyCommand] = {}
    for run in payload.get("runs", []):
        prompt = normalize_prompt(run.get("prompt_index") or run.get("prompt_id"))
        prompt_id = clean_text(run.get("prompt_id"))
        for row in run.get("commands", []):
            command_index = coerce_int(row.get("command_index") or row.get("photo_match_index"))
            if command_index is None:
                continue
            commands[(prompt, command_index)] = PolicyCommand(
                prompt=prompt,
                prompt_id=prompt_id,
                command_index=command_index,
                command=clean_text(row.get("command")),
                thought=clean_text(row.get("raw_policy_thought")),
                reason=clean_text(row.get("reason")),
                timestamp_local=clean_text(row.get("timestamp_local")),
            )
    if not commands:
        raise SystemExit(f"ERROR: no commands found in policy JSON: {policy_json}")
    return commands


def selected_policy_commands(
    commands: dict[tuple[str, int], PolicyCommand],
    *,
    important_commands: set[str],
    all_commands: bool,
) -> list[PolicyCommand]:
    rows = sorted(commands.values(), key=lambda item: (int(item.prompt or 0), item.command_index))
    if all_commands:
        return rows
    return [row for row in rows if row.command in important_commands]


def write_manifest_template(
    path: Path,
    commands: dict[tuple[str, int], PolicyCommand],
    *,
    important_commands: set[str],
    all_commands: bool,
    populate_text: bool,
) -> None:
    rows: list[dict[str, str]] = []
    for command in selected_policy_commands(commands, important_commands=important_commands, all_commands=all_commands):
        rows.append(
            {
                "include": "yes",
                "prompt": command.prompt,
                "command_index": str(command.command_index),
                "image_path": "",
                "command": command.command if populate_text else "",
                "thought": command.thought if populate_text else "",
                "note": f"Policy reason: {command.reason}" if populate_text and command.reason else "",
                "video_path": "",
                "video_time_s": "",
                "tile_title": f"Prompt {command.prompt} - Command {command.command_index}" if populate_text else "",
            }
        )

    rows.extend(
        [
            {
                "include": "no",
                "prompt": "4",
                "command_index": "",
                "image_path": "",
                "command": "",
                "thought": "No raw policy thought captured for Prompt 4.",
                "note": "Prompt 4 example: add an exported video frame path here, then set include=yes.",
                "video_path": "",
                "video_time_s": "",
                "tile_title": "Prompt 4 video frame",
            },
            {
                "include": "no",
                "prompt": "4",
                "command_index": "",
                "image_path": "",
                "command": "",
                "thought": "No raw policy thought captured for Prompt 4.",
                "note": "Prompt 4 example: use this for another key frame from the video.",
                "video_path": "",
                "video_time_s": "",
                "tile_title": "Prompt 4 video frame",
            },
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"ERROR: manifest CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise SystemExit(f"ERROR: manifest CSV has no header: {path}")
        missing = [field for field in MANIFEST_FIELDS if field not in reader.fieldnames]
        if missing:
            raise SystemExit(f"ERROR: manifest is missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def safe_asset_name(source: Path, *, prompt: str, command_index: str, row_number: int) -> str:
    suffix = source.suffix.lower() or ".jpg"
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    prompt_part = re.sub(r"[^A-Za-z0-9_-]+", "_", prompt or "prompt")
    command_part = re.sub(r"[^A-Za-z0-9_-]+", "_", command_index or f"row{row_number:03d}")
    return f"{prompt_part}_{command_part}_{digest}{suffix}"


def copy_image_asset(
    image_path: str,
    *,
    manifest_dir: Path,
    assets_dir: Path,
    prompt: str,
    command_index: str,
    row_number: int,
) -> tuple[str, str, str]:
    image_text = clean_text(image_path)
    if not image_text:
        return "", "missing_path", ""
    source = resolve_path(image_text, base_dir=manifest_dir)
    if not source.exists() or not source.is_file():
        return "", "missing_file", str(source)
    assets_dir.mkdir(parents=True, exist_ok=True)
    name = safe_asset_name(source, prompt=prompt, command_index=command_index, row_number=row_number)
    destination = assets_dir / name
    shutil.copy2(source, destination)
    return f"assets/{name}", "ok", str(source)


def build_tiles(
    manifest_path: Path,
    commands: dict[tuple[str, int], PolicyCommand],
    *,
    output_dir: Path,
    max_thought_chars: int,
) -> list[Tile]:
    manifest_dir = manifest_path.parent
    assets_dir = output_dir / "assets"
    tiles: list[Tile] = []
    for row_number, row in enumerate(read_manifest(manifest_path), start=2):
        if not truthy_include(row.get("include", "")):
            continue
        prompt = normalize_prompt(row.get("prompt"))
        command_index_int = coerce_int(row.get("command_index"))
        command_index = str(command_index_int) if command_index_int is not None else clean_text(row.get("command_index"))
        policy = commands.get((prompt, command_index_int or -1))

        command = clean_text(row.get("command")) or (policy.command if policy else "")
        thought = clean_text(row.get("thought")) or (policy.thought if policy else "")
        if not thought and prompt == "4":
            thought = "No raw policy thought captured for Prompt 4."
        note = clean_text(row.get("note"))
        tile_title = clean_text(row.get("tile_title"))
        if not tile_title:
            if command_index:
                tile_title = f"Prompt {prompt} - Command {command_index}"
            else:
                tile_title = f"Prompt {prompt}"

        image_src, image_status, source_image = copy_image_asset(
            row.get("image_path", ""),
            manifest_dir=manifest_dir,
            assets_dir=assets_dir,
            prompt=prompt,
            command_index=command_index,
            row_number=row_number,
        )
        if not command and not note:
            note = f"No policy command matched manifest row {row_number}; fill command/thought manually."

        tiles.append(
            Tile(
                prompt=prompt,
                command_index=command_index,
                title=tile_title,
                command=command or "manual / video frame",
                thought=trim_text(thought or "No thought text supplied.", max_thought_chars),
                note=note,
                image_src=image_src,
                image_status=image_status,
                source_image=source_image,
                video_path=clean_text(row.get("video_path")),
                video_time_s=clean_text(row.get("video_time_s")),
            )
        )
    if not tiles:
        raise SystemExit(f"ERROR: manifest has no included rows: {manifest_path}")
    return tiles


def html_text(value: object) -> str:
    return html.escape(str(value or ""), quote=True).replace("\n", "<br>")


def render_tile(tile: Tile) -> str:
    if tile.image_status == "ok":
        image_html = f'<img class="tile-image" src="{html_text(tile.image_src)}" alt="{html_text(tile.title)}">'
    elif tile.image_status == "missing_file":
        image_html = (
            '<div class="placeholder">'
            "<strong>Image file missing</strong>"
            f"<span>{html_text(tile.source_image)}</span>"
            "</div>"
        )
    else:
        image_html = (
            '<div class="placeholder">'
            "<strong>No image yet</strong>"
            "<span>Fill image_path in the manifest when ready.</span>"
            "</div>"
        )

    note_html = f'<p class="note"><strong>Note:</strong> {html_text(tile.note)}</p>' if tile.note else ""
    video_html = ""
    if tile.video_path or tile.video_time_s:
        pieces = []
        if tile.video_path:
            pieces.append(f"Video: {tile.video_path}")
        if tile.video_time_s:
            pieces.append(f"Time: {tile.video_time_s}s")
        video_html = f'<p class="video-note">{html_text(" | ".join(pieces))}</p>'

    return f"""
      <article class="tile">
        <header>
          <div class="eyebrow">Prompt {html_text(tile.prompt)}</div>
          <h3>{html_text(tile.title)}</h3>
        </header>
        <div class="image-wrap">{image_html}</div>
        <p class="command"><strong>Command:</strong> <code>{html_text(tile.command)}</code></p>
        <p class="thought"><strong>Thought:</strong> {html_text(tile.thought)}</p>
        {note_html}
        {video_html}
      </article>
"""


def render_html(
    tiles: list[Tile],
    *,
    title: str,
    manifest_path: Path,
    policy_json: Path,
) -> str:
    groups: dict[str, list[Tile]] = {}
    for tile in tiles:
        groups.setdefault(tile.prompt, []).append(tile)

    sections: list[str] = []
    for prompt in sorted(groups, key=lambda value: int(value) if value.isdigit() else 999):
        cards = "\n".join(render_tile(tile) for tile in groups[prompt])
        sections.append(
            f"""
    <section class="prompt-section">
      <h2>Prompt {html_text(prompt)}</h2>
      <div class="grid">
        {cards}
      </div>
    </section>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(title)}</title>
  <style>
    @page {{
      size: Letter;
      margin: 0.45in;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      color: #111827;
      background: #f3f4f6;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 11px;
      line-height: 1.35;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    .doc-title {{
      margin: 0 0 4px;
      font-size: 28px;
      line-height: 1.1;
      color: #0f172a;
    }}
    .meta {{
      margin: 0 0 18px;
      color: #4b5563;
      font-size: 12px;
    }}
    .prompt-section {{
      margin: 24px 0 36px;
      break-before: auto;
    }}
    .prompt-section + .prompt-section {{
      break-before: page;
    }}
    h2 {{
      margin: 0 0 10px;
      color: #1f4d78;
      font-size: 20px;
      line-height: 1.2;
      border-bottom: 2px solid #dbeafe;
      padding-bottom: 5px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }}
    .tile {{
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px;
      break-inside: avoid;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    .tile header {{
      margin-bottom: 7px;
    }}
    .eyebrow {{
      color: #64748b;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 2px;
    }}
    h3 {{
      margin: 0;
      color: #0f172a;
      font-size: 14px;
      line-height: 1.2;
    }}
    .image-wrap {{
      width: 100%;
      aspect-ratio: 4 / 3;
      border: 1px solid #e5e7eb;
      background: #f8fafc;
      margin-bottom: 8px;
      overflow: hidden;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .tile-image {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #f8fafc;
    }}
    .placeholder {{
      color: #64748b;
      text-align: center;
      padding: 16px;
    }}
    .placeholder strong {{
      display: block;
      color: #334155;
      margin-bottom: 5px;
      font-size: 13px;
    }}
    .placeholder span {{
      display: block;
      overflow-wrap: anywhere;
    }}
    p {{
      margin: 0 0 6px;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 11px;
      color: #111827;
      background: #f1f5f9;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    .thought {{
      color: #1f2937;
    }}
    .note, .video-note {{
      color: #6b7280;
      font-size: 10.5px;
    }}
    @media print {{
      body {{
        background: white;
        font-size: 9.5px;
      }}
      main {{
        max-width: none;
        padding: 0;
      }}
      .doc-title {{
        font-size: 22px;
      }}
      .meta {{
        font-size: 10px;
      }}
      h2 {{
        font-size: 16px;
      }}
      h3 {{
        font-size: 11.5px;
      }}
      .grid {{
        gap: 8px;
      }}
      .tile {{
        box-shadow: none;
        border-radius: 5px;
        padding: 8px;
      }}
      .image-wrap {{
        margin-bottom: 6px;
      }}
      code {{
        font-size: 9.5px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1 class="doc-title">{html_text(title)}</h1>
    <p class="meta">
      Manifest: {html_text(str(manifest_path))}<br>
      Policy JSON: {html_text(str(policy_json))}<br>
      Tiles: {len(tiles)}
    </p>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def write_collage(
    manifest_path: Path,
    policy_json: Path,
    output_dir: Path,
    *,
    title: str,
    max_thought_chars: int,
) -> Path:
    commands = load_policy_commands(policy_json)
    tiles = build_tiles(
        manifest_path,
        commands,
        output_dir=output_dir,
        max_thought_chars=max_thought_chars,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(
        render_html(tiles, title=title, manifest_path=manifest_path, policy_json=policy_json),
        encoding="utf-8",
    )
    return html_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-json", default=DEFAULT_POLICY_JSON, help="Policy thought export JSON.")
    parser.add_argument("--manifest", default="", help="CSV selecting which tiles to show.")
    parser.add_argument("--write-template", default="", help="Write an editable manifest CSV template and exit unless --manifest is also supplied.")
    parser.add_argument("--template-all-commands", action="store_true", help="Template includes every command instead of only important moves.")
    parser.add_argument(
        "--template-populate-text",
        action="store_true",
        help="Fill command/thought/note/title columns in the generated manifest; only image_path stays blank.",
    )
    parser.add_argument(
        "--important-commands",
        default=",".join(DEFAULT_IMPORTANT_COMMANDS),
        help="Comma-separated commands included in the default template.",
    )
    parser.add_argument("--output-dir", default="results/policy_collage", help="Directory for index.html and copied image assets.")
    parser.add_argument("--title", default="Policy Photo Collage", help="HTML title and page heading.")
    parser.add_argument(
        "--max-thought-chars",
        type=int,
        default=520,
        help="Trim thought text for tile readability; use 0 for full thought text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = project_root()
    policy_json = resolve_path(args.policy_json)
    commands = load_policy_commands(policy_json)
    important_commands = {
        item.strip()
        for item in str(args.important_commands).split(",")
        if item.strip()
    }

    if args.write_template:
        template_path = resolve_path(args.write_template)
        write_manifest_template(
            template_path,
            commands,
            important_commands=important_commands,
            all_commands=bool(args.template_all_commands),
            populate_text=bool(args.template_populate_text),
        )
        print(f"Wrote manifest template: {template_path}")
        if not args.manifest:
            return 0

    if not args.manifest:
        raise SystemExit(
            "ERROR: --manifest is required to build the collage. "
            "Use --write-template results/policy_collage_manifest_template.csv first."
        )

    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    html_path = write_collage(
        manifest_path,
        policy_json,
        output_dir,
        title=str(args.title),
        max_thought_chars=int(args.max_thought_chars),
    )
    try:
        rel_html = html_path.relative_to(root)
    except ValueError:
        rel_html = html_path
    print(f"Wrote collage HTML: {rel_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
