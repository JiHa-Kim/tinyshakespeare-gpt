import json
import math
from pathlib import Path


def jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class MetricsLogger:
    def __init__(self, path: str, run_name: str = "") -> None:
        self.run_name = run_name
        self.file = None
        if path:
            metrics_path = Path(path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = metrics_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **values) -> None:
        if self.file is None:
            return
        record = {"event": event, "run_name": self.run_name, **values}
        self.file.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
