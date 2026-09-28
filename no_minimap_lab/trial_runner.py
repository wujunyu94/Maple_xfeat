"""GUI launcher for the unchanged one-round coverage CLI."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from .atlas import ROOT
from .input_service import STATE


def service_ready():
    try:
        state = json.loads((STATE/'service.json').read_text(encoding='utf-8'))
        return bool(state.get('ready') and not state.get('stopped') and psutil.pid_exists(state['pid']))
    except (OSError, ValueError, KeyError):
        return False


def command(output):
    return [sys.executable, '-m', 'no_minimap_lab.coverage_trial',
            '--backend', 'xfeat', '--rounds', '1', '--output', str(output)]


class TrialRunner:
    def __init__(self):
        self.process = None
        self.output = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        if self.running:
            raise RuntimeError('完整测试已在运行')
        for process in psutil.process_iter(['pid', 'cmdline']):
            if 'no_minimap_lab.coverage_trial' in (process.info.get('cmdline') or []):
                raise RuntimeError('检测到已有完整测试进程，请先等待它结束')
        if not service_ready():
            raise RuntimeError('按键服务未就绪。请先点击“启动按键服务（管理员）”，完成 UAC 后再开始。')
        self.output = ROOT/'no_minimap_lab'/'output'/('retest_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.output.mkdir(parents=True)
        (STATE/'stop_navigation').unlink(missing_ok=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        with (self.output/'console.log').open('w',encoding='utf-8') as log:
            self.process = subprocess.Popen(command(self.output), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return self.output

    def stop(self):
        if self.running:
            (STATE/'stop_navigation').write_text('GUI requested stop',encoding='utf-8')

    def progress(self):
        if self.output is None:
            return '尚未开始'
        try:
            progress = json.loads((self.output/'round_01'/'progress.json').read_text(encoding='utf-8'))
            event = progress.get('event','')
            observation = progress.get('observation') or {}
            platform = observation.get('platform')
            labels = dict(start='前往目标平台', movement_begin='移动对齐', movement_control='移动控制',
                          movement_arrived='已对齐', action_attempt='执行动作', action_result='动作完成',
                          tour_start='开始覆盖全部绳梯', coverage_target='前往下一条绳梯',
                          coverage_target_result='绳梯测试完成', edge_start='开始平台移动',
                          edge_end='平台移动完成', grab_verified='已确认抓稳绳梯')
            return labels.get(event,event)+(f' · P{platform}' if platform else '')
        except (OSError, ValueError):
            return '正在加载地图、模型或生成报告'


def launch_service():
    if service_ready():
        return
    import ctypes
    # Explicit GUI button invokes UAC; service only exposes restricted movement.
    result = ctypes.windll.shell32.ShellExecuteW(None,'runas',sys.executable,
        subprocess.list2cmdline(['-m','no_minimap_lab.input_service']),str(ROOT),0)
    if result <= 32:
        raise RuntimeError('按键服务未启动（UAC 被取消或系统拒绝）')
