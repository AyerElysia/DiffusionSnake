import json
import os


class JsonLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # line-buffered append; write/flush failures are deliberate fail-fast errors.
        self._f = open(self.path, 'a', buffering=1, encoding='utf-8')

    def log(self, obj: dict):
        payload = json.dumps(obj, ensure_ascii=False, allow_nan=False)
        self._f.write(payload + "\n")
        self._f.flush()

    def close(self):
        self._f.close()

    def tell(self):
        self._f.flush()
        return os.path.getsize(self.path)

    def truncate(self, pos):
        if pos is None:
            return
        self._f.flush()
        pos = int(pos)
        current_size = os.path.getsize(self.path)
        if pos < 0 or pos > current_size:
            raise ValueError(f'invalid JSONL truncate position {pos} for size {current_size}')
        self._f.truncate(pos)
        self._f.flush()


class NullLogger:
    def log(self, obj: dict):
        return

    def close(self):
        return

    def tell(self):
        return None

    def truncate(self, pos):
        return
