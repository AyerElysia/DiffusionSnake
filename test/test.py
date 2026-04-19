import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.argv.extend([
    '--cfg_file',
    str(PROJECT_ROOT / 'configs' / 'sbd_snake.yaml'),
    'ct_score',
    '0.4',
    'train_or_test',
    'test',
])

from run import run_test_medical
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
run_test_medical()
