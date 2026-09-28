"""Reusable visual diagnostics: bounded FPS video with exact frame timestamps.

Own this object on the capture/vision worker. Recording does not issue inputs or
perform localization. Call close() on shutdown; the JSONL index survives an
interruption and identifies the captured time of every submitted video frame.
"""
import json
from pathlib import Path
import cv2


class DiagnosticRecorder:
    def __init__(self, output, fps=10):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        if fps<=0:
            raise ValueError('fps must be positive')
        self.fps=fps
        self.next_time=0
        self.writer=None
        self.index=None
        self.frames=0

    def record(self, frame, packet):
        tick=packet['time']
        if tick<self.next_time:
            return False
        self.next_time=tick+1/self.fps
        if self.writer is None:
            self.writer=cv2.VideoWriter(str(self.output/'replay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),
                                       self.fps,(frame.shape[1],frame.shape[0]))
            if not self.writer.isOpened():
                self.writer.release();self.writer=None
                raise RuntimeError('Diagnostic video writer could not open')
            self.index=(self.output/'video_frames.jsonl').open('w',encoding='utf-8',buffering=1)
        marked=frame.copy()
        cv2.putText(marked,f"{packet['status']} P{packet['platform']} {packet['world']}",
                    (12,60),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
        self.writer.write(marked)
        self.index.write(json.dumps(dict(frame=self.frames,capture_time=tick,
            observed_at=packet.get('observed_at'),playback_seconds=self.frames/self.fps))+'\n')
        self.frames+=1
        cv2.imwrite(str(self.output/'latest.jpg'),marked)
        (self.output/'live.json').write_text(json.dumps(packet),encoding='utf-8')
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release();self.writer=None
        if self.index is not None:
            self.index.close();self.index=None
