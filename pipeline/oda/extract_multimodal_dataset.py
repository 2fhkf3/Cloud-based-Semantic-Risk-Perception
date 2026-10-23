#!/usr/bin/env python3
"""Extract RGB-frame/state pairs from the ODA dataset.

Extended from the version shipped with ``multimodal_output/trial_3``:

- ``trial_overview.csv`` rows are grouped per Sequence, so trials flown with two
  obstacles keep both obstacles (the original only read the first matching row).
- Trials that cannot be processed (missing obstacle coordinates, missing or
  corrupt video) raise a :class:`TrialSkipError` subclass so the batch driver in
  ``extract_multimodal_batch.py`` can record and skip them.

The generated four-direction ranges are geometric labels, not measurements read
from radar.csv.  They are computed from the OptiTrack pose, the obstacle
metadata, and optional rectangular wall bounds.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "Missing dependency 'opencv-python'. Install it with: "
        "python -m pip install opencv-python"
    ) from exc


EPSILON = 1.0e-12
Vector2 = Tuple[float, float]
Quaternion = Tuple[float, float, float, float]
Obstacle = Tuple[float, float, float]


class TrialSkipError(RuntimeError):
    """Raised when a trial cannot be processed and should be skipped."""


class MissingVideoError(TrialSkipError):
    pass


class MissingObstacleCoordinatesError(TrialSkipError):
    pass


@dataclass(frozen=True)
class PoseSample:
    time_s: float
    raw_x: float
    raw_y: float
    raw_z: float
    quaternion_xyzw: Quaternion

    @property
    def position_xy(self) -> Vector2:
        # ODA's official visualization uses raw x/z as the horizontal plane and
        # raw y as height.
        return self.raw_x, self.raw_z


class PoseSeries:
    def __init__(self, samples: Sequence[PoseSample]) -> None:
        if not samples:
            raise ValueError("optitrack.csv contains no pose samples")
        self.samples = list(samples)
        self.times = [sample.time_s for sample in samples]

    @property
    def start_time_s(self) -> float:
        return self.times[0]

    @property
    def end_time_s(self) -> float:
        return self.times[-1]

    def interpolate(self, time_s: float) -> PoseSample:
        if time_s <= self.times[0]:
            return self.samples[0]
        if time_s >= self.times[-1]:
            return self.samples[-1]

        right = bisect.bisect_left(self.times, time_s)
        left = right - 1
        before = self.samples[left]
        after = self.samples[right]
        interval = after.time_s - before.time_s
        alpha = 0.0 if interval <= EPSILON else (time_s - before.time_s) / interval

        return PoseSample(
            time_s=time_s,
            raw_x=lerp(before.raw_x, after.raw_x, alpha),
            raw_y=lerp(before.raw_y, after.raw_y, alpha),
            raw_z=lerp(before.raw_z, after.raw_z, alpha),
            quaternion_xyzw=nlerp_quaternion(
                before.quaternion_xyzw, after.quaternion_xyzw, alpha
            ),
        )


def lerp(start: float, end: float, alpha: float) -> float:
    return start + alpha * (end - start)


def normalize_quaternion(q: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= EPSILON:
        raise ValueError("encountered a zero-length OptiTrack quaternion")
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def nlerp_quaternion(q0: Quaternion, q1: Quaternion, alpha: float) -> Quaternion:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    # q and -q represent the same rotation. Select the short interpolation arc.
    if sum(a * b for a, b in zip(q0, q1)) < 0.0:
        q1 = tuple(-component for component in q1)  # type: ignore[assignment]
    blended = tuple(lerp(a, b, alpha) for a, b in zip(q0, q1))
    return normalize_quaternion(blended)  # type: ignore[arg-type]


def rotate_vector_by_quaternion(
    vector: Tuple[float, float, float], quaternion: Quaternion
) -> Tuple[float, float, float]:
    """Rotate a 3-D vector by a normalized (x, y, z, w) quaternion."""
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    vx, vy, vz = vector

    # v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def body_forward_vector(axis: str) -> Tuple[float, float, float]:
    vectors = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    return vectors[axis]


def quaternion_yaw(
    quaternion: Quaternion,
    forward_axis: str,
    initial_y_rotation_offset: float,
    offset_mode: str,
) -> float:
    forward_world = rotate_vector_by_quaternion(
        body_forward_vector(forward_axis), quaternion
    )
    # Horizontal ODA coordinates are raw world X and raw world Z.
    yaw = math.atan2(forward_world[2], forward_world[0])
    if offset_mode == "subtract":
        yaw -= initial_y_rotation_offset
    elif offset_mode == "add":
        yaw += initial_y_rotation_offset
    return wrap_angle(yaw)


def trajectory_yaw(series: PoseSeries, time_s: float, window_s: float) -> Optional[float]:
    before_t = max(series.start_time_s, time_s - window_s)
    after_t = min(series.end_time_s, time_s + window_s)
    before = series.interpolate(before_t).position_xy
    after = series.interpolate(after_t).position_xy
    dx = after[0] - before[0]
    dy = after[1] - before[1]
    if math.hypot(dx, dy) <= 1.0e-6:
        return None
    return math.atan2(dy, dx)


def load_optitrack(path: Path) -> PoseSeries:
    samples: List[PoseSample] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                PoseSample(
                    time_s=float(row["time [ns]"]) * 1.0e-9,
                    raw_x=float(row["x [m]"]),
                    raw_y=float(row["y [m]"]),
                    raw_z=float(row["z [m]"]),
                    quaternion_xyzw=(
                        float(row["a [rad]"]),
                        float(row["b [rad]"]),
                        float(row["c [rad]"]),
                        float(row["d [rad]"]),
                    ),
                )
            )
    return PoseSeries(samples)


def load_overview_groups(path: Path) -> Dict[int, List[Dict[str, str]]]:
    """Group every ``trial_overview.csv`` row by its Sequence number.

    Two-obstacle trials appear as two consecutive rows with the same Sequence;
    keeping all of them preserves both obstacles.
    """
    groups: Dict[int, List[Dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence = (row.get("Sequence") or "").strip()
            if not sequence:
                continue
            groups.setdefault(int(float(sequence)), []).append(row)
    return groups


def load_trial_obstacles(
    rows: Sequence[Dict[str, str]], trial_id: int
) -> Tuple[List[Obstacle], float]:
    """Return ``[(x, z, radius), ...]`` in ODA horizontal coordinates plus the
    initial yaw offset, using every overview row of this trial."""
    obstacles: List[Obstacle] = []
    initial_offset = float(rows[0]["OptiTrack initial y rotation offset"])
    for row in rows:
        raw_x = (row.get("Obstacle x") or "").strip()
        raw_z = (row.get("Obstacle z") or "").strip()
        if not raw_x or not raw_z:
            raise MissingObstacleCoordinatesError(
                f"trial {trial_id} has no obstacle coordinates in trial_overview.csv"
            )
        obstacles.append((float(raw_x), float(raw_z)))
    return obstacles, initial_offset


def ray_circle_distance(
    origin: Vector2, direction: Vector2, center: Vector2, radius: float
) -> float:
    vx = center[0] - origin[0]
    vy = center[1] - origin[1]
    center_distance_sq = vx * vx + vy * vy
    if center_distance_sq <= radius * radius:
        return 0.0

    projection = vx * direction[0] + vy * direction[1]
    if projection < 0.0:
        return math.inf

    perpendicular_sq = max(0.0, center_distance_sq - projection * projection)
    if perpendicular_sq > radius * radius:
        return math.inf

    entry = projection - math.sqrt(max(0.0, radius * radius - perpendicular_sq))
    return entry if entry >= 0.0 else math.inf


def ray_rectangle_distance(
    origin: Vector2, direction: Vector2, bounds: Tuple[float, float, float, float]
) -> float:
    xmin, xmax, ymin, ymax = bounds
    px, py = origin
    ux, uy = direction
    candidates: List[float] = []

    if abs(ux) > EPSILON:
        for wall_x in (xmin, xmax):
            distance = (wall_x - px) / ux
            hit_y = py + distance * uy
            if distance >= 0.0 and ymin - EPSILON <= hit_y <= ymax + EPSILON:
                candidates.append(distance)

    if abs(uy) > EPSILON:
        for wall_y in (ymin, ymax):
            distance = (wall_y - py) / uy
            hit_x = px + distance * ux
            if distance >= 0.0 and xmin - EPSILON <= hit_x <= xmax + EPSILON:
                candidates.append(distance)

    return min(candidates, default=math.inf)


def cardinal_directions(yaw: float) -> List[Tuple[str, Vector2]]:
    angles = [
        ("front", yaw),
        ("left", yaw + math.pi / 2.0),
        ("back", yaw + math.pi),
        ("right", yaw - math.pi / 2.0),
    ]
    return [(name, (math.cos(angle), math.sin(angle))) for name, angle in angles]


def raycast_ranges(
    position: Vector2,
    yaw: float,
    obstacles: Sequence[Obstacle],
    wall_bounds: Optional[Tuple[float, float, float, float]],
    max_range_m: float,
) -> Tuple[List[float], List[str]]:
    distances: List[float] = []
    hits: List[str] = []
    for _, direction in cardinal_directions(yaw):
        best_distance = max_range_m
        best_hit = "max_range"

        for index, (cx, cy, radius) in enumerate(obstacles):
            distance = ray_circle_distance(position, direction, (cx, cy), radius)
            if distance < best_distance:
                best_distance = distance
                best_hit = f"obstacle_{index}"

        if wall_bounds is not None:
            wall_distance = ray_rectangle_distance(position, direction, wall_bounds)
            if wall_distance < best_distance:
                best_distance = wall_distance
                best_hit = "wall"

        distances.append(min(max_range_m, max(0.0, best_distance)))
        hits.append(best_hit)
    return distances, hits


def sector_ranges(
    position: Vector2,
    yaw: float,
    obstacles: Sequence[Obstacle],
    wall_bounds: Optional[Tuple[float, float, float, float]],
    max_range_m: float,
) -> Tuple[List[float], List[str]]:
    """Assign obstacles to four 90-degree sectors, then take nearest surfaces."""
    distances = [max_range_m, max_range_m, max_range_m, max_range_m]
    hits = ["max_range", "max_range", "max_range", "max_range"]

    for index, (cx, cy, radius) in enumerate(obstacles):
        dx = cx - position[0]
        dy = cy - position[1]
        relative = wrap_angle(math.atan2(dy, dx) - yaw)
        if -math.pi / 4.0 <= relative < math.pi / 4.0:
            direction_index = 0  # front
        elif math.pi / 4.0 <= relative < 3.0 * math.pi / 4.0:
            direction_index = 1  # left
        elif -3.0 * math.pi / 4.0 <= relative < -math.pi / 4.0:
            direction_index = 3  # right
        else:
            direction_index = 2  # back

        surface_distance = max(0.0, math.hypot(dx, dy) - radius)
        if surface_distance < distances[direction_index]:
            distances[direction_index] = surface_distance
            hits[direction_index] = f"obstacle_{index}"

    # Walls are still measured along the four central rays.
    if wall_bounds is not None:
        for direction_index, (_, direction) in enumerate(cardinal_directions(yaw)):
            wall_distance = ray_rectangle_distance(position, direction, wall_bounds)
            if wall_distance < distances[direction_index]:
                distances[direction_index] = wall_distance
                hits[direction_index] = "wall"

    return [min(max_range_m, value) for value in distances], hits


def rounded(value: float, decimals: int) -> float:
    result = round(float(value), decimals)
    return 0.0 if result == -0.0 else result


def validate_wall_bounds(
    values: Optional[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
    if values is None:
        return None
    xmin, xmax, ymin, ymax = (float(value) for value in values)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("wall bounds must satisfy xmin < xmax and ymin < ymax")
    return xmin, xmax, ymin, ymax


@dataclass
class ExtractOptions:
    """Per-trial extraction parameters shared by the CLI and the batch driver."""

    range_method: str = "raycast"
    range_max_m: float = 4.0
    obstacle_radius_m: float = 0.2
    inflate_obstacles_by_m: float = 0.0
    extra_obstacle: Optional[List[Tuple[float, float, float]]] = None
    wall_bounds: Optional[Tuple[float, float, float, float]] = None
    goal_xy: Optional[Tuple[float, float]] = None
    yaw_source: str = "quaternion"
    forward_axis: str = "+x"
    rotation_offset_mode: str = "subtract"
    trajectory_yaw_window_s: float = 0.15
    fps_override: Optional[float] = None
    sample_every: int = 1
    jpeg_quality: int = 95
    numeric_decimals: int = 6
    timestamp_decimals: int = 2
    skip_images: bool = False


def extract_trial(
    trial_id: int,
    dataset_root: Path,
    overview_groups: Dict[int, List[Dict[str, str]]],
    output_dir: Path,
    options: ExtractOptions,
) -> Dict[str, object]:
    """Extract one trial's frames and geometry labels; returns its metadata.

    Raises ``TrialSkipError`` (or ``FileNotFoundError``) for trials that must be
    skipped; other exceptions indicate real bugs.
    """
    if options.range_max_m <= 0.0:
        raise ValueError("--range-max-m must be positive")
    if options.obstacle_radius_m <= 0.0:
        raise ValueError("--obstacle-radius-m must be positive")
    if options.inflate_obstacles_by_m < 0.0:
        raise ValueError("--inflate-obstacles-by-m cannot be negative")
    if options.sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if not 1 <= options.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    dataset_root = dataset_root.resolve()
    trial_dir = dataset_root / str(trial_id)
    video_path = trial_dir / f"{trial_id}.avi"
    optitrack_path = trial_dir / "optitrack.csv"
    if not video_path.is_file():
        raise MissingVideoError(f"trial {trial_id} has no {video_path.name}")
    if not optitrack_path.is_file():
        raise FileNotFoundError(optitrack_path)

    rows = overview_groups.get(trial_id)
    if not rows:
        raise MissingObstacleCoordinatesError(
            f"trial {trial_id} was not found in trial_overview.csv"
        )
    overview_positions, initial_offset = load_trial_obstacles(rows, trial_id)

    effective_radius = options.obstacle_radius_m + options.inflate_obstacles_by_m
    obstacles: List[Obstacle] = [(x, z, effective_radius) for x, z in overview_positions]
    if options.extra_obstacle:
        for x, y, radius in options.extra_obstacle:
            if radius <= 0.0:
                raise ValueError("extra obstacle radius must be positive")
            obstacles.append((x, y, radius + options.inflate_obstacles_by_m))

    wall_bounds = validate_wall_bounds(options.wall_bounds)
    poses = load_optitrack(optitrack_path)
    if options.goal_xy is None:
        goal_xy = poses.samples[-1].position_xy
        goal_source = "trajectory_endpoint"
    else:
        goal_xy = (float(options.goal_xy[0]), float(options.goal_xy[1]))
        goal_source = "explicit"

    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    annotations_path = output_dir / "annotations.jsonl"
    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not options.skip_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise MissingVideoError(f"OpenCV could not open {video_path}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = float(options.fps_override) if options.fps_override is not None else reported_fps
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise MissingVideoError(
            f"trial {trial_id} video FPS is unavailable; provide fps_override explicitly"
        )

    range_function = raycast_ranges if options.range_method == "raycast" else sector_ranges
    range_geometry = (
        "obstacles_and_walls" if wall_bounds is not None else "known_obstacles_only"
    )
    frame_index = 0
    exported_count = 0
    used_names = set()
    last_yaw: Optional[float] = None

    try:
        with annotations_path.open("w", encoding="utf-8", newline="\n") as annotations:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % options.sample_every != 0:
                    frame_index += 1
                    continue

                timestamp_s = frame_index / fps
                pose = poses.interpolate(timestamp_s)
                position_xy = pose.position_xy

                if options.yaw_source == "quaternion":
                    yaw = quaternion_yaw(
                        pose.quaternion_xyzw,
                        options.forward_axis,
                        initial_offset,
                        options.rotation_offset_mode,
                    )
                else:
                    yaw = trajectory_yaw(
                        poses, timestamp_s, options.trajectory_yaw_window_s
                    )
                    if yaw is None:
                        yaw = last_yaw
                    if yaw is None:
                        yaw = quaternion_yaw(
                            pose.quaternion_xyzw,
                            options.forward_axis,
                            initial_offset,
                            options.rotation_offset_mode,
                        )
                last_yaw = yaw

                goal_distance = math.hypot(
                    goal_xy[0] - position_xy[0], goal_xy[1] - position_xy[1]
                )
                directional_ranges, range_hits = range_function(
                    position_xy,
                    yaw,
                    obstacles,
                    wall_bounds,
                    options.range_max_m,
                )

                timestamp_label = f"{timestamp_s:.{options.timestamp_decimals}f}"
                image_name = f"trial_{trial_id}_t{timestamp_label}.jpg"
                if image_name in used_names:
                    image_name = (
                        f"trial_{trial_id}_frame_{frame_index:06d}_"
                        f"t{timestamp_label}.jpg"
                    )
                used_names.add(image_name)
                relative_image_path = Path("images") / image_name
                if not options.skip_images:
                    destination = output_dir / relative_image_path
                    wrote_image = cv2.imwrite(
                        str(destination),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), options.jpeg_quality],
                    )
                    if not wrote_image:
                        raise RuntimeError(f"failed to write {destination}")

                state = [
                    position_xy[0],
                    position_xy[1],
                    goal_distance,
                    directional_ranges[0],
                    directional_ranges[1],
                    directional_ranges[2],
                    directional_ranges[3],
                ]
                record = {
                    "image": relative_image_path.as_posix(),
                    "trial_id": trial_id,
                    "frame_index": frame_index,
                    "timestamp_s": rounded(timestamp_s, options.numeric_decimals),
                    "position_xy": [
                        rounded(position_xy[0], options.numeric_decimals),
                        rounded(position_xy[1], options.numeric_decimals),
                    ],
                    "goal_xy": [
                        rounded(goal_xy[0], options.numeric_decimals),
                        rounded(goal_xy[1], options.numeric_decimals),
                    ],
                    "goal_source": goal_source,
                    "yaw_rad": rounded(yaw, options.numeric_decimals),
                    "state": [rounded(value, options.numeric_decimals) for value in state],
                    "state_valid": [1, 1, 1, 1, 1, 1, 1],
                    "range_method": options.range_method,
                    "range_max_m": options.range_max_m,
                    "range_geometry": range_geometry,
                    "wall_bounds_provided": wall_bounds is not None,
                    "range_hits": range_hits,
                    "pose_time_clamped": not (
                        poses.start_time_s <= timestamp_s <= poses.end_time_s
                    ),
                }
                annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported_count += 1
                frame_index += 1
    finally:
        capture.release()

    if exported_count == 0:
        raise MissingVideoError(f"trial {trial_id} video decoded to zero frames")

    metadata = {
        "trial_id": trial_id,
        "video": str(video_path),
        "video_fps": fps,
        "decoded_video_frames": frame_index,
        "exported_frames": exported_count,
        "sample_every": options.sample_every,
        "annotations": str(annotations_path),
        "images": None if options.skip_images else str(image_dir),
        "coordinate_mapping": {
            "horizontal_x": "optitrack raw x",
            "horizontal_y": "optitrack raw z",
            "height": "optitrack raw y",
        },
        "goal_xy": list(goal_xy),
        "goal_source": goal_source,
        "yaw": {
            "source": options.yaw_source,
            "body_forward_axis": options.forward_axis,
            "initial_y_rotation_offset_rad": initial_offset,
            "rotation_offset_mode": options.rotation_offset_mode,
            "trajectory_window_s": options.trajectory_yaw_window_s,
        },
        "range": {
            "method": options.range_method,
            "max_m": options.range_max_m,
            "geometry": range_geometry,
            "wall_bounds_xy": list(wall_bounds) if wall_bounds is not None else None,
            "obstacles_xyr": [list(obstacle) for obstacle in obstacles],
            "obstacle_radius_m": options.obstacle_radius_m,
            "inflate_obstacles_by_m": options.inflate_obstacles_by_m,
            "source": "synthetic geometry; not radar measurements",
        },
        "synchronization": {
            "method": "video frame_index / fps, then OptiTrack interpolation",
            "limitation": "ODA CSV streams independently start at relative time zero",
        },
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    return metadata


def build_parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract per-RGB-frame ODA image/state JSONL data for one trial."
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="directory containing trial_overview.csv and <id>/ trial dirs")
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--range-method", choices=("raycast", "sector"), default="raycast")
    parser.add_argument("--range-max-m", type=float, default=4.0)
    parser.add_argument("--obstacle-radius-m", type=float, default=0.2)
    parser.add_argument(
        "--inflate-obstacles-by-m",
        type=float,
        default=0.0,
        help="Optional extra radius, e.g. drone safety radius.",
    )
    parser.add_argument(
        "--extra-obstacle",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "RADIUS"),
        help="Add another horizontal-plane circular obstacle; may be repeated.",
    )
    parser.add_argument(
        "--wall-bounds",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional rectangular wall bounds in ODA horizontal X/Z coordinates.",
    )
    parser.add_argument(
        "--goal-xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Explicit goal; default is the last OptiTrack horizontal position.",
    )
    parser.add_argument(
        "--yaw-source", choices=("quaternion", "trajectory"), default="quaternion"
    )
    parser.add_argument(
        "--forward-axis", choices=("+x", "-x", "+z", "-z"), default="+x"
    )
    parser.add_argument(
        "--rotation-offset-mode",
        choices=("subtract", "add", "ignore"),
        default="subtract",
        help="How to apply trial_overview's initial Y rotation offset.",
    )
    parser.add_argument("--trajectory-yaw-window-s", type=float, default=0.15)
    parser.add_argument("--fps", type=float, help="Override FPS reported by the AVI file.")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--numeric-decimals", type=int, default=6)
    parser.add_argument("--timestamp-decimals", type=int, default=2)
    parser.add_argument("--skip-images", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    overview_groups = load_overview_groups(dataset_root / "trial_overview.csv")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_root.parent / "multimodal_output" / f"trial_{args.trial_id}"
    )
    options = ExtractOptions(
        range_method=args.range_method,
        range_max_m=args.range_max_m,
        obstacle_radius_m=args.obstacle_radius_m,
        inflate_obstacles_by_m=args.inflate_obstacles_by_m,
        extra_obstacle=(
            [tuple(item) for item in args.extra_obstacle]
            if args.extra_obstacle
            else None
        ),
        wall_bounds=(
            tuple(args.wall_bounds) if args.wall_bounds is not None else None
        ),
        goal_xy=tuple(args.goal_xy) if args.goal_xy is not None else None,
        yaw_source=args.yaw_source,
        forward_axis=args.forward_axis,
        rotation_offset_mode=args.rotation_offset_mode,
        trajectory_yaw_window_s=args.trajectory_yaw_window_s,
        fps_override=args.fps,
        sample_every=args.sample_every,
        jpeg_quality=args.jpeg_quality,
        numeric_decimals=args.numeric_decimals,
        timestamp_decimals=args.timestamp_decimals,
        skip_images=args.skip_images,
    )
    metadata = extract_trial(
        args.trial_id, dataset_root, overview_groups, output_dir, options
    )
    print(f"Exported {metadata['exported_frames']} records to {output_dir / 'annotations.jsonl'}")
    print(f"Images: {'skipped' if args.skip_images else output_dir / 'images'}")
    print(f"Range geometry: {metadata['range']['geometry']}")
    if metadata["range"]["wall_bounds_xy"] is None:
        print(
            "Warning: no wall bounds were supplied; direction ranges only use the "
            "known obstacle metadata and max range.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TrialSkipError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
