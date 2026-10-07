"""OS file dialogs run in a separate process so Tk owns its main thread."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['image', 'folder'], required=True)
    parser.add_argument('--initial-dir', default='')
    args = parser.parse_args()
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        initial = Path(args.initial_dir).expanduser()
        options = {'parent': root, 'initialdir': str(initial) if initial.is_dir() else str(Path.home())}
        if args.kind == 'folder':
            chosen = filedialog.askdirectory(title='Choose an OCT image folder', mustexist=True, **options)
        else:
            chosen = filedialog.askopenfilename(title='Choose an OCT TIFF volume',
                filetypes=[('TIFF volumes', '*.tif *.tiff *.TIF *.TIFF'), ('All files', '*')], **options)
        print(json.dumps({'path': chosen or None}))
    except Exception:
        print(json.dumps({'error': 'The system file chooser could not open. Use the folder path field instead. A graphical desktop and Python Tk support are required.'}))
    finally:
        if root is not None:
            root.destroy()


if __name__ == '__main__':
    main()
