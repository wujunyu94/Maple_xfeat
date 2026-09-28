"""Check dependencies before importing the GUI's third-party modules."""
import argparse
import runpy
import sys
from .check_environment import check


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend',default='sift-cpu')
    args,_=parser.parse_known_args()
    report=check(args.backend)
    if not report['ok']:
        message='运行环境检查未通过：\n'+'\n'.join(report['errors'])
        print(message)
        if not any(flag in sys.argv for flag in ('--image','--video','--live')):
            try:
                import tkinter as tk
                from tkinter import messagebox
                root=tk.Tk();root.withdraw()
                messagebox.showerror('启动检查',message,parent=root);root.destroy()
            except Exception:
                pass
        return 1
    runpy.run_module('no_minimap_lab.run',run_name='__main__')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
