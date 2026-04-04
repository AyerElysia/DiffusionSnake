import json
import os


class JsonLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # line-buffered append
        self._f = open(self.path, 'a', buffering=1, encoding='utf-8')

    def log(self, obj: dict):
        try:
            self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def tell(self):
        try:
            self._f.flush()
            return os.path.getsize(self.path)
        except Exception:
            return None

    def truncate(self, pos):
        if pos is None:
            return
        try:
            self._f.flush()
            pos = int(pos)
            current_size = os.path.getsize(self.path)
            if pos < 0 or pos > current_size:
                return
            self._f.truncate(pos)
        except Exception:
            pass


class NullLogger:
    def log(self, obj: dict):
        return

    def close(self):
        return

    def tell(self):
        return None

    def truncate(self, pos):
        return
