import argparse
import pathlib
import sys


FORBIDDEN_TERMS = (
    "pseudo3d",
    "three_frame",
    "three_slice_input",
    "neighbor_mean",
    "neighbor_fusion",
    "prev_contour",
    "previous_contour",
    "video_frame",
    "temporal_memory",
    "true_3d",
)

ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".md"}


def iter_files(paths):
    for path in paths:
        path = pathlib.Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in ALLOWED_SUFFIXES:
                    yield child


def main():
    parser = argparse.ArgumentParser(
        description="Check VolMem naming and maturity contracts"
    )
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    violations = []
    for path in iter_files(args.paths):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        relative = str(path).lower()
        if path.name == "check_contract.py":
            continue
        for term in FORBIDDEN_TERMS:
            if term in relative or term in text:
                violations.append("{}: forbidden term {!r}".format(path, term))

    if violations:
        for violation in violations:
            print(violation)
        return 1

    print("VolMem contract check passed for {} path(s)".format(len(args.paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
