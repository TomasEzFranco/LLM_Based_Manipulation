#!/usr/bin/env python3
import os

import cv2
import numpy as np

# --- YOLO (Ultralytics). If you have a different detector, wire it in below. ---
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _env_bool(name: str, default: bool) -> bool:
    default_text = "1" if default else "0"
    return os.getenv(name, default_text).strip().lower() in {"1", "true", "yes", "on"}


# Hard-lock tracker backend to Bot-SORT + ReID.
# QARM_YOLO_TRACKER is intentionally ignored/deprecated in this mode.
YOLO_TRACKER = "trackers/botsort_reid.yaml"
YOLO_TRACK_PERSIST = _env_bool("QARM_YOLO_TRACK_PERSIST", True)

# Locked detect profile (vision_runtime is source of truth; runtime_core imports these).
# Tune duplicate/overlap behavior with Ultralytics NMS IoU only.
YOLO_IOU_NMS = float(os.getenv("QARM_YOLO_IOU_NMS", "0.5"))
YOLO_STRETCH_SQUARE = _env_bool("QARM_YOLO_STRETCH_SQUARE", True)
YOLO_STRETCH_SIZE = int(os.getenv("QARM_YOLO_STRETCH_SIZE", "640"))


class YOLODetector:
    def __init__(self, path, target_classes=None, conf=0.35, tracker: str = YOLO_TRACKER, track_persist: bool = YOLO_TRACK_PERSIST):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install `ultralytics` or swap in your detector.")
        self.model = YOLO(path)
        self.target_class_ids = set()
        self.target_class_names = set()
        for target in (target_classes or ()):
            if isinstance(target, int):
                self.target_class_ids.add(target)
            elif isinstance(target, str):
                text = target.strip()
                if text.isdigit():
                    self.target_class_ids.add(int(text))
                self.target_class_names.add(text.lower())
        self.conf = conf
        tracker_norm = str(tracker or "").strip().lower()
        if tracker_norm and tracker_norm != str(YOLO_TRACKER).strip().lower():
            raise ValueError(
                f"Unsupported tracker override '{tracker}'. "
                f"Runtime is hard-locked to '{YOLO_TRACKER}'."
            )
        self.tracker = str(YOLO_TRACKER)
        self.track_persist = bool(track_persist)
        self._track_recovery_count = 0

    def _is_target(self, cls: int | None, name: str | None) -> bool:
        if not self.target_class_ids and not self.target_class_names:
            return True
        if cls is not None and cls in self.target_class_ids:
            return True
        if name is not None and name.lower() in self.target_class_names:
            return True
        return False

    def detect_candidates_and_draw(self, bgr, draw: bool = True):
        img_display = bgr.copy() if bool(draw) else bgr
        orig_h, orig_w = int(bgr.shape[0]), int(bgr.shape[1])
        detect_iou = float(YOLO_IOU_NMS)
        if bool(YOLO_STRETCH_SQUARE):
            sz = max(1, int(YOLO_STRETCH_SIZE))
            src = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_LINEAR)
            scale_x = float(orig_w) / float(sz)
            scale_y = float(orig_h) / float(sz)
            infer_imgsz = int(sz)
        else:
            src = bgr
            scale_x = 1.0
            scale_y = 1.0
            infer_imgsz = int(src.shape[1])
        track_kwargs = {
            "source": src,
            "imgsz": infer_imgsz,
            "conf": self.conf,
            "iou": detect_iou,
            "verbose": False,
            "persist": bool(self.track_persist),
            "stream": False,
        }
        if self.tracker:
            track_kwargs["tracker"] = self.tracker
        using_predict_no_track = False
        try:
            results = self.model.track(**track_kwargs)
        except Exception as exc:
            # Tracker can fail intermittently; recover once before failing loudly.
            # Recover once with persist=False, then fall back to one-frame predict.
            msg = str(exc)
            self._track_recovery_count = int(self._track_recovery_count) + 1
            print(
                "[YOLOTrackRecover] tracker failure; "
                f"retrying frame with persist=False (count={int(self._track_recovery_count)}): {msg}"
            )
            retry_kwargs = dict(track_kwargs)
            retry_kwargs["persist"] = False
            try:
                results = self.model.track(**retry_kwargs)
            except Exception as retry_exc:
                predict_kwargs = {
                    "source": src,
                    "imgsz": infer_imgsz,
                    "conf": self.conf,
                    "iou": detect_iou,
                    "verbose": False,
                    "stream": False,
                }
                print(
                    "[YOLOTrackFallback] using predict_no_track "
                    f"(initial={msg}; retry={retry_exc})"
                )
                try:
                    results = self.model.predict(**predict_kwargs)
                    using_predict_no_track = True
                except Exception as predict_exc:
                    raise RuntimeError(
                        "YOLO detect failed after track retry and predict fallback "
                        f"(initial={msg}; retry={retry_exc}; predict={predict_exc})"
                    ) from predict_exc
        target_candidates = []
        if results:
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0]) if b.cls is not None else None
                    name = self.model.names.get(cls, None) if hasattr(self.model, "names") else None
                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    track_id = None
                    if not bool(using_predict_no_track):
                        try:
                            if getattr(b, "id", None) is not None:
                                track_id = int(float(b.id[0]))
                        except Exception:
                            track_id = None
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    if scale_x != 1.0 or scale_y != 1.0:
                        x1 = int(np.clip(round(float(x1) * scale_x), 0, max(0, orig_w - 1)))
                        y1 = int(np.clip(round(float(y1) * scale_y), 0, max(0, orig_h - 1)))
                        x2 = int(np.clip(round(float(x2) * scale_x), 0, max(0, orig_w - 1)))
                        y2 = int(np.clip(round(float(y2) * scale_y), 0, max(0, orig_h - 1)))
                        if x2 < x1:
                            x1, x2 = x2, x1
                        if y2 < y1:
                            y1, y2 = y2, y1
                    is_target = self._is_target(cls, name)
                    if draw:
                        color = (0, 255, 0) if is_target else (255, 0, 0)  # Green target, Blue ignored
                        thickness = 3 if is_target else 2
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)
                        label = f"{name if name else cls}: {conf:.2f}"
                        if track_id is not None:
                            label += f" id={track_id}"
                        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(img_display, (x1, y1 - label_h - baseline - 5),
                                    (x1 + label_w, y1), color, -1)
                        cv2.putText(img_display, label, (x1, y1 - baseline - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if is_target:
                        u = int(0.5 * (x1 + x2))
                        v = int(0.875 * 0.5 * (y1 + y2))  # top-face bias for better depth sample
                        row = {
                            "u": u,
                            "v": v,
                            "conf": conf,
                            "cls": cls,
                            "name": name,
                            "bbox_xyxy": (x1, y1, x2, y2),
                            "track_id": track_id,
                            "is_tracked": bool((not using_predict_no_track) and track_id is not None),
                        }
                        target_candidates.append(row)
        return img_display, target_candidates

    def select_closest_to_center(
        self,
        candidates,
        cx: int,
        cy: int,
        min_conf: float = 0.0,
        return_candidate: bool = False,
    ):
        best = None
        best_key = None
        best_candidate = None
        for candidate in candidates:
            conf = float(candidate["conf"])
            if conf < min_conf:
                continue
            u = int(candidate["u"])
            v = int(candidate["v"])
            d2 = (u - cx) * (u - cx) + (v - cy) * (v - cy)
            key = (d2, -conf)
            if best is None or key < best_key:
                best = (u, v, conf)
                best_key = key
                best_candidate = candidate
        if return_candidate:
            return best_candidate
        return best


def _finite_base_z_from_uv(arm, per, depth_frame, u: int, v: int) -> float:
    from projection_geometry import estimate_base_xyz_from_uv_fast

    xyz = estimate_base_xyz_from_uv_fast(
        arm=arm,
        per=per,
        depth_frame=depth_frame,
        u=int(u),
        v=int(v),
    )
    if xyz.size < 3 or not np.isfinite(float(xyz[2])):
        return float("nan")
    return float(xyz[2])


def _split_one_merged_candidate(
    row: dict,
    *,
    arm,
    per,
    depth_frame,
    layer_dz_m: float,
    min_height_m: float,
    min_aspect: float,
    max_cubes: int,
) -> list[dict]:
    raw_bbox = row.get("bbox_xyxy", None)
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return [dict(row)]
    try:
        x1 = int(float(raw_bbox[0]))
        y1 = int(float(raw_bbox[1]))
        x2 = int(float(raw_bbox[2]))
        y2 = int(float(raw_bbox[3]))
    except (TypeError, ValueError):
        return [dict(row)]
    if x2 <= x1 or y2 <= y1:
        return [dict(row)]

    bw = int(x2 - x1)
    bh = int(y2 - y1)
    aspect = float(bh) / float(max(1, bw))
    if aspect < float(min_aspect):
        return [dict(row)]

    if arm is None or per is None or depth_frame is None:
        return [dict(row)]

    u_mid = int(0.5 * (x1 + x2))
    z_top = _finite_base_z_from_uv(arm, per, depth_frame, u_mid, y1)
    z_bot = _finite_base_z_from_uv(arm, per, depth_frame, u_mid, y2)
    if not np.isfinite(z_top) or not np.isfinite(z_bot):
        return [dict(row)]
    delta_z = float(z_top - z_bot)
    if delta_z < float(min_height_m):
        return [dict(row)]

    dz_step = max(1e-6, float(layer_dz_m))
    n_cubes = int(round(delta_z / dz_step))
    n_cubes = max(2, min(int(max_cubes), int(n_cubes)))

    parent_u = int(row.get("u", u_mid))
    parent_v = int(row.get("v", int(0.875 * 0.5 * (y1 + y2))))
    parent_tid = row.get("track_id", None)
    parent_tracked = bool(row.get("is_tracked", False))

    children: list[dict] = []
    child_uvs: list[tuple[int, int]] = []
    for k in range(int(n_cubes)):
        y1_k = int(y2 - int((k + 1) * bh / n_cubes))
        y2_k = int(y2 - int(k * bh / n_cubes))
        if y2_k <= y1_k:
            y2_k = int(y1_k + 1)
        target_z = float(z_bot + (float(k) + 0.5) * dz_step)
        frac = float((target_z - z_bot) / max(1e-6, delta_z))
        frac = float(max(0.0, min(1.0, frac)))
        v_k = int(y2 - frac * float(bh))
        v_k = int(max(y1_k, min(y2_k - 1, v_k)))
        v_k = int(0.875 * float(v_k) + 0.125 * float(y1_k))
        u_k = int(u_mid)

        child = {
            "u": int(u_k),
            "v": int(v_k),
            "conf": float(row.get("conf", 0.0)),
            "cls": row.get("cls", None),
            "name": row.get("name", None),
            "bbox_xyxy": (int(x1), int(y1_k), int(x2), int(y2_k)),
            "track_id": None,
            "is_tracked": False,
            "source": "split_from_merged",
        }
        children.append(child)
        child_uvs.append((int(u_k), int(v_k)))

    if not children:
        return [dict(row)]

    best_idx = 0
    best_d2 = float("inf")
    for idx, child in enumerate(children):
        du = int(child["u"]) - int(parent_u)
        dv = int(child["v"]) - int(parent_v)
        d2 = float(du * du + dv * dv)
        if d2 < best_d2:
            best_d2 = d2
            best_idx = int(idx)
    if parent_tid is not None:
        try:
            children[int(best_idx)]["track_id"] = int(parent_tid)
            children[int(best_idx)]["is_tracked"] = bool(parent_tracked)
        except (TypeError, ValueError):
            pass

    print(
        f"[BboxSplit] orig=({x1},{y1},{x2},{y2}) delta_z={delta_z:.3f} "
        f"n_cubes={int(n_cubes)} children_uv={child_uvs}"
    )
    return children


def split_merged_stack_candidates(
    candidates: list[dict],
    *,
    arm,
    per,
    depth_frame,
    layer_dz_m: float,
    min_height_m: float,
    min_aspect: float,
    max_cubes: int,
) -> list[dict]:
    try:
        if not candidates:
            return list(candidates)
        out: list[dict] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            out.extend(
                _split_one_merged_candidate(
                    row,
                    arm=arm,
                    per=per,
                    depth_frame=depth_frame,
                    layer_dz_m=float(layer_dz_m),
                    min_height_m=float(min_height_m),
                    min_aspect=float(min_aspect),
                    max_cubes=int(max_cubes),
                )
            )
        return out if out else list(candidates)
    except Exception as exc:
        print(f"[BboxSplit] error={exc}; passthrough original candidates")
        return list(candidates)
