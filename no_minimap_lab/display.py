import cv2
import numpy as np

from .localizer import scene_mask


def annotate(frame, model, pose, player=None):
    out = frame.copy()
    mask = scene_mask(frame.shape)
    out[mask == 0] = (out[mask == 0]*.25).astype(np.uint8)
    if pose.camera is not None and model.track_points is not None:
        for x, y in model.track_points:
            point = (round(float(x)), round(float(y)))
            cv2.circle(out, point, 5, (15, 45, 20), -1)
            cv2.circle(out, point, 4, (60, 245, 80), -1)
    if player:
        x, y = map(round, player["screen"])
        predicted = player.get("status") == "PREDICTED"
        cv2.drawMarker(out, (x, y), (0, 140, 255) if predicted else (0, 230, 255), cv2.MARKER_CROSS, 20, 2)
        if predicted:
            cv2.circle(out, (x, y), min(150, round(player.get("uncertainty_px", 20)*model.screen_scale)), (0, 140, 255), 1)
        if player.get("feature_bbox"):
            fx, fy, fw, fh = player["feature_bbox"]
            cv2.rectangle(out, (fx, fy), (fx+fw, fy+fh), (255, 180, 20), 2)
    color = (70, 255, 90) if pose.status == "LOCKED" else (0, 190, 255)
    label = f"{pose.status}  landmarks={pose.inliers} / rival={pose.runner_up}  {pose.elapsed_ms:.0f}ms"
    cv2.putText(out, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    if pose.camera:
        cv2.putText(out, f"Camera world: ({pose.camera[0]:.1f}, {pose.camera[1]:.1f})",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1)
    return out


def overview(atlas, frame_shape, pose, screen_scale, player=None, size=(950, 360), yellow=None, background=None):
    h, w = atlas.bgr.shape[:2]
    scale = min(size[0]/w, size[1]/h)
    out = background.copy() if background is not None else cv2.resize(atlas.bgr, (max(1, round(w*scale)), max(1, round(h*scale))))
    def point(p):
        return tuple(np.round((np.array(p)-atlas.origin)*scale).astype(int))
    for node in ([] if background is not None else atlas.meta.get("ladder_nodes", [])):
        color = (220, 120, 240) if node["kind"] == "rope" else (60, 180, 255)
        cv2.line(out, point((node["x"], node["y1"])), point((node["x"], node["y2"])), color, 1)
    if pose.camera:
        camera = np.array(pose.camera)
        far = camera + np.array(frame_shape[:2][::-1])/screen_scale
        cv2.rectangle(out, point(camera), point(far), (40, 220, 255), 2)
        if player:
            world = camera + np.array(player["screen"])/screen_scale
            predicted = player.get('status') == 'PREDICTED'
            color = (0, 150, 255) if predicted else (20, 20, 255)
            cv2.circle(out, point(world), 6, color, -1)
            cv2.circle(out, point(world), max(7, round(player.get('uncertainty_px', 3)*scale)), color, 1)
            if player.get('velocity') is not None:
                future = world+np.array(player['velocity'])*.15
                cv2.arrowedLine(out, point(world), point(future), (0, 150, 255), 2)
                cv2.drawMarker(out, point(future), (0, 150, 255), cv2.MARKER_DIAMOND, 12, 1)
    if yellow and yellow.get("detected"):
        cv2.drawMarker(out, point(yellow["raw_world"]), (255, 240, 0), cv2.MARKER_CROSS, 14, 2)
    return out


def add_ladder_overlay(image, advice, ladders=None):
    if ladders is not None:
        for node in ladders:
            verified = node["status"] in ("verified", "recently_verified")
            color = (90, 245, 110) if verified else (0, 180, 255)
            dx = node["correction_x_px"] if verified else 0
            for x, y1, y2 in node["segments"]:
                x, y1, y2 = round(x+dx), round(y1), round(y2)
                if verified:
                    cv2.line(image, (x, y1), (x, y2), color, 2)
                else:
                    for y in range(y1, y2, 12):
                        cv2.line(image, (x, y), (x, min(y+6, y2)), color, 2)
                cv2.line(image, (x-6, y1), (x+6, y1), color, 2)
                cv2.line(image, (x-6, y2), (x+6, y2), color, 2)
            x, y, _ = node["segments"][0]
            label = f"{'L' if node['kind']=='ladder' else 'R'}#{node['id']}"
            label += f" {node['score']:.2f}" if verified else " MAP"
            tx, ty = min(image.shape[1]-100, round(x+9)), max(18, round(y+15))
            cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
            cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
    if advice:
        x, y1, y2 = map(round, advice["screen_line"])
        if ladders is None:
            cv2.line(image, (x, y1), (x, y2), (255, 120, 200), 2)
        cv2.putText(image, f"WZ ladder dx={advice['dx_screen']:+.1f}px (projection)",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 120, 200), 1)
    return image
