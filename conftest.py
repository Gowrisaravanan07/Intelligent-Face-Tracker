import sys
import os

# Add root project directory to sys.path for test discovery
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure temp directory uses D: drive with abundant disk space
tmp_dir = os.path.join(root_dir, "scratch", "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir
os.environ["TEMP"] = tmp_dir
os.environ["TMP"] = tmp_dir

