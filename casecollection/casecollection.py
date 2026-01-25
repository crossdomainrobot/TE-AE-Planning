import os
import sys
import re
import json
import argparse
import ast
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Iterable, Set
from collections import deque

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_A = "qwen-plus"
MODEL_B = "qwen-plus"

API_KEY_A = ""
API_KEY_B = ""
MAP_FILE_PATH = r"D:\Pycharm\Platoon\mapgeneration\30_30\30_30.txt"

if not API_KEY_A:
    raise RuntimeError(
        "Missing API key. Set env var DASHSCOPE_API_KEY_A (and optionally DASHSCOPE_API_KEY_B). "
        "If you already have literal keys in your local file, keep them there (do not paste them here)."
    )
if not API_KEY_B:
    API_KEY_B = API_KEY_A

client_a = OpenAI(api_key=API_KEY_A, base_url=BASE_URL)
client_b = OpenAI(api_key=API_KEY_B, base_url=BASE_URL)


SYSTEM_PROMPT_A = (
    """Terrestrial-aerial cross-domain robotics possesses cross-domain mobility capabilities, 
enabling it to fly at different altitudes as well as travel on the ground. Compared with aerial flight, 
ground locomotion generally incurs significantly lower energy consumption. You act as a path-planning 
guide for terrestrial-aerial cross-domain robotics. Based on a given map matrix, which represents the
spatial grid layout of the environment, and an obstacle height matrix, which specifies the height of
obstacles at each grid cell, you must perform structured reasoning that jointly considers feasibility, 
safety, and energy efficiency. Your task is not to generate a complete path, but to infer and output
exactly five guiding regions that you consider reasonable and informative, explicitly providing their
corresponding grid coordinates. A guiding region refers to a grid location through which a path is more
likely to pass in order to achieve a globally lower-cost solution, particularly by favoring ground traversal,
reducing energy consumption, and avoiding high or impassable obstacles. The detailed definition, value semantics,
and coordinate conventions of the map matrix are described below.

Two two-dimensional matrices of identical size are provided to represent a grid-based map environment.
The first matrix is an occupancy grid, whose elements take only the values 0 or 1,
where 0 indicates that the corresponding cell is free space and directly traversable,
and 1 indicates that the cell contains an obstacle and is not traversable.
The second matrix is a height map, which is spatially aligned with the occupancy grid on a one-to-one basis,
and is used to describe the height information of obstacles at each grid cell.
When the corresponding cell in the occupancy grid is 0, the height value is typically 0, indicating no obstacle,
and when the corresponding cell is 1, the height value is a positive number,
representing the height level or relative vertical magnitude of the obstacle.
Together, the two matrices provide a joint representation of the environment's two-dimensional spatial structure
and the vertical characteristics of obstacles.

Please think step by step and reason carefully. First, analyze the meaning of the map matrix and the obstacle
height matrix that I will provide next. Then, consider which locations are more likely to guide the CDR toward
a path with lower energy consumption, and determine the corresponding guiding region positions. Finally, output
a total of five guiding regions. Note that you must also take directional guidance into account: the five guiding
regions should be progressively closer to the goal, where the start point is the upper-left corner and the goal is
the lower-right corner."""
)

SYSTEM_PROMPT_B = (
    "You are an information extractor responsible for extracting"
    "the positions of five guiding regions from my input. Note"
    "that in your output you must not include any information"
    "other than the separator “,” and the coordinates of the five"
    "guiding regions. You only need to output exactly five coordinates,"
    "separated by commas."
)
def resolve_map_file(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists() and p.is_file():
        return p

    for ext in (".text", ".txt"):
        p2 = Path(str(p) + ext)
        if p2.exists() and p2.is_file():
            return p2

    raise FileNotFoundError(f"Map file not found: {path_str}")


def _read_text_smart(file_path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _extract_braced_block(text: str, start_brace_pos: int) -> Tuple[str, int]:
    if start_brace_pos < 0 or start_brace_pos >= len(text) or text[start_brace_pos] != "{":
        raise ValueError("start_brace_pos must point to '{'")

    depth = 0
    in_str = False
    str_quote = ""
    escape = False

    i = start_brace_pos
    while i < len(text):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == str_quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                str_quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_brace_pos : i + 1], i + 1
        i += 1

    raise ValueError("Unbalanced braces: cannot find matching '}'")


def load_maps_from_textfile(txtfile: Path) -> Dict[int, Dict[str, Any]]:
    text = _read_text_smart(txtfile)

    stripped = text.lstrip()
    if stripped.startswith("{") and ("\"map0\"" in stripped or "\"map1\"" in stripped or "\"map" in stripped):
        try:
            obj = json.loads(stripped)
            maps: Dict[int, Dict[str, Any]] = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    m = re.fullmatch(r"map(\d+)", str(k))
                    if m and isinstance(v, dict):
                        maps[int(m.group(1))] = v
            if maps:
                return maps
        except Exception:
            pass

    maps: Dict[int, Dict[str, Any]] = {}
    pattern = re.compile(r"\bmap(\d+)\s*=", re.MULTILINE)
    for m in pattern.finditer(text):
        idx = int(m.group(1))
        eq_end = m.end()
        brace_pos = text.find("{", eq_end)
        if brace_pos == -1:
            continue

        try:
            block, _end = _extract_braced_block(text, brace_pos)
        except Exception:
            continue

        try:
            d = ast.literal_eval(block)
            if isinstance(d, dict):
                maps[idx] = d
        except Exception:
            continue

    if not maps:
        preview = text[:300].replace("\n", "\\n")
        raise RuntimeError(
            f"Failed to parse any maps from text file: {txtfile}\n"
            f"File head preview (300 chars): {preview}"
        )

    return maps


def get_map_matrices(maps: Dict[int, Dict[str, Any]], map_index: int) -> Tuple[Any, Any]:
    if map_index not in maps:
        available = sorted(maps.keys())
        raise KeyError(
            f"map{map_index} not found. Available indices (first 20): {available[:20]} (total {len(available)})"
        )

    m = maps[map_index]
    if "data" not in m or "height" not in m:
        raise KeyError(f"map{map_index} must contain keys 'data' and 'height'.")

    return m["data"], m["height"]


def build_user_input_with_matrices(data_matrix: Any, height_matrix: Any) -> str:
    data_str = json.dumps(data_matrix, ensure_ascii=False)
    height_str = json.dumps(height_matrix, ensure_ascii=False)

    user_input = (
        "Below are the map occupancy matrix (0=free, 1=obstacle) and the obstacle height matrix.\n\n"
        f"Map matrix (occupancy grid):\n{data_str}\n\n"
        f"Height matrix:\n{height_str}\n\n"
        "Above is the map and the height matrix I provided. "
        "First, think about what this matrix means. I need you to "
        "provide me with the coordinates of five ‘guidance regions’ "
        "that you consider reasonable (i.e., you think a path that passes "
        "through these regions could have lower energy consumption). But "
        "also note that these five guidance regions should be progressively "
        "closer to the destination: that is, for each subsequent point, its "
        "projection onto the line connecting the start and the destination should "
        "be closer to the destination. The start is the top-left corner, and the "
        "destination is the bottom-right corner. Please think step by step, slowly."
    )
    return user_input


def iter_stream_text(stream) -> Iterable[str]:
    for event in stream:
        try:
            if hasattr(event, "choices") and event.choices:
                delta = getattr(event.choices[0], "delta", None)
                if delta is not None:
                    chunk = getattr(delta, "content", None)
                    if chunk:
                        yield chunk
        except Exception:
            continue


def stream_chat_completion(client: OpenAI, model: str, messages: list, print_prefix: str = "") -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    full_text_parts: List[str] = []
    for chunk in iter_stream_text(stream):
        full_text_parts.append(chunk)
        if print_prefix:
            sys.stdout.write(print_prefix)
            print_prefix = ""
        sys.stdout.write(chunk)
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full_text_parts)


_NEI8 = [(-1, -1), (0, -1), (1, -1),
         (-1,  0),          (1,  0),
         (-1,  1), (0,  1), (1,  1)]


def _to_float_matrix(mat: Any) -> List[List[float]]:
    if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
        raise ValueError("height_matrix must be a 2D list.")
    out: List[List[float]] = []
    for row in mat:
        out.append([float(v) for v in row])
    return out


def _infer_rc_from_coord(coord: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
    a, b = coord
    xy_ok = (0 <= a < cols) and (0 <= b < rows)
    rc_ok = (0 <= a < rows) and (0 <= b < cols)

    if xy_ok and not rc_ok:
        return (b, a)
    if rc_ok and not xy_ok:
        return (a, b)
    if xy_ok and rc_ok:
        return (b, a)

    raise ValueError(f"Coordinate {coord} out of bounds for map (rows={rows}, cols={cols}).")


def _free_at_h(height_mat: List[List[float]], r: int, c: int, h: float) -> bool:
    return h >= height_mat[r][c]


def _bfs_from_start(height_mat: List[List[float]], h: float,
                    start_rc: Tuple[int, int]) -> Set[Tuple[int, int]]:
    rows, cols = len(height_mat), len(height_mat[0])
    sr, sc = start_rc
    if not _free_at_h(height_mat, sr, sc, h):
        return set()

    q = deque([(sr, sc)])
    vis: Set[Tuple[int, int]] = {(sr, sc)}

    while q:
        r, c = q.popleft()
        for dc, dr in _NEI8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in vis:
                continue
            if not _free_at_h(height_mat, nr, nc, h):
                continue
            vis.add((nr, nc))
            q.append((nr, nc))
    return vis


def _connected_via_at_h(height_mat: List[List[float]], h: float,
                        via_rc: Tuple[int, int],
                        start_rc: Tuple[int, int],
                        goal_rc: Tuple[int, int]) -> bool:
    vr, vc = via_rc
    sr, sc = start_rc
    gr, gc = goal_rc

    if not _free_at_h(height_mat, vr, vc, h):
        return False
    if not _free_at_h(height_mat, gr, gc, h):
        return False
    if not _free_at_h(height_mat, sr, sc, h):
        return False

    vis = _bfs_from_start(height_mat, h, start_rc)
    return (via_rc in vis) and (goal_rc in vis)


def min_connected_height_for_point(height_mat: List[List[float]],
                                   via_rc: Tuple[int, int],
                                   start_rc: Tuple[int, int],
                                   goal_rc: Tuple[int, int]) -> Optional[float]:
    uniq: Set[float] = set()
    for row in height_mat:
        for v in row:
            uniq.add(float(v))
    candidates = sorted(uniq)

    if not candidates:
        return None

    lo, hi = 0, len(candidates) - 1
    ans_idx: Optional[int] = None

    if not _connected_via_at_h(height_mat, candidates[hi], via_rc, start_rc, goal_rc):
        return None

    while lo <= hi:
        mid = (lo + hi) // 2
        h = candidates[mid]
        if _connected_via_at_h(height_mat, h, via_rc, start_rc, goal_rc):
            ans_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return candidates[ans_idx] if ans_idx is not None else None


def parse_five_coords_from_b_output(b_output: str) -> List[Tuple[int, int]]:
    matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", b_output)
    if len(matches) != 5:
        raise ValueError(f"Expected exactly 5 coordinates in B output, got {len(matches)}. Raw: {b_output!r}")
    coords = [(int(a), int(b)) for a, b in matches]
    return coords


def main():
    parser = argparse.ArgumentParser(description="Map-indexed A->B prompting script + minimal connectivity heights.")
    parser.add_argument(
        "--map_index",
        type=int,
        default=None,
        help="地图索引，例如 10 表示读取 map10 的 data 与 height。",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=MAP_FILE_PATH,
        help="地图文件路径（.text/.txt 或无后缀）。",
    )
    args = parser.parse_args()

    if args.map_index is None:
        try:
            raw = input("Input  map index：").strip()
            args.map_index = int(raw)
        except Exception as e:
            raise RuntimeError("Invalid map_index input. Please input an integer.") from e

    map_file = resolve_map_file(args.map_path)
    print(f"Using map text file: {map_file}")

    maps = load_maps_from_textfile(map_file)

    data_matrix, height_matrix_raw = get_map_matrices(maps, args.map_index)
    height_matrix = _to_float_matrix(height_matrix_raw)

    rows, cols = len(height_matrix), len(height_matrix[0])
    start_rc = (0, 0)
    goal_rc = (rows - 1, cols - 1)

    user_input = build_user_input_with_matrices(data_matrix, height_matrix_raw)

    messages_a = [
        {"role": "system", "content": SYSTEM_PROMPT_A},
        {"role": "user", "content": user_input},
    ]

    print("========== Model A (stream) ==========")
    a_output = stream_chat_completion(
        client=client_a,
        model=MODEL_A,
        messages=messages_a,
        print_prefix="A> ",
    )

    b_user_content = (
        "Below is the output from Model A. Use it as input and produce the final result.\n\n"
        "=== Model A Output Start ===\n"
        f"{a_output}\n"
        "=== Model A Output End ==="
    )

    messages_b = [
        {"role": "system", "content": SYSTEM_PROMPT_B},
        {"role": "user", "content": b_user_content},
    ]

    print("\n========== Model B (stream) ==========")
    b_output = stream_chat_completion(
        client=client_b,
        model=MODEL_B,
        messages=messages_b,
        print_prefix="B> ",
    )

    coords = parse_five_coords_from_b_output(b_output)

    per_point_min_h: List[Optional[float]] = []
    per_point_rc: List[Tuple[int, int]] = []

    for coord in coords:
        rc = _infer_rc_from_coord(coord, rows=rows, cols=cols)
        per_point_rc.append(rc)
        hmin = min_connected_height_for_point(height_matrix, rc, start_rc, goal_rc)
        per_point_min_h.append(hmin)

    valid_h = [h for h in per_point_min_h if h is not None]
    total_sum = sum(valid_h)

    print("\n========== Minimal 8-connected heights (start -> via AND start -> goal) ==========")
    print(f"Map size: rows={rows}, cols={cols}")
    print(f"Start (row,col) = {start_rc}, Goal (row,col) = {goal_rc}")
    print("Note: B output coords are inferred as (x,y) by default when ambiguous; converted to (row,col) for computation.\n")

    for i, (coord, rc, hmin) in enumerate(zip(coords, per_point_rc, per_point_min_h), start=1):
        print(f"[{i}] B_coord={coord} -> rc={rc} : min_connected_height = {hmin}")

    print("\nSum of minimal connected heights (excluding None) =", total_sum)


if __name__ == "__main__":
    main()
