import argparse
import pathlib
import sys


ALLOWED_TRAINING_MATURITY = {"prototype", "baseline", "validated"}


def read_maturity(config_path):
    text = pathlib.Path(config_path).read_text(encoding="utf-8")
    in_volmem = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "volmem:":
            in_volmem = True
            continue
        if in_volmem and line.startswith("maturity:"):
            return line.split(":", 1)[1].strip()
    raise ValueError("volmem.maturity is required")


def main():
    parser = argparse.ArgumentParser(description="VolMemSnake training entry")
    parser.add_argument("--cfg_file", required=True)
    args = parser.parse_args()

    maturity = read_maturity(args.cfg_file)
    if maturity not in ALLOWED_TRAINING_MATURITY:
        raise RuntimeError(
            "Refusing to train VolMem config with maturity={!r}. Complete the "
            "data contract, memory forward path and smoke tests first.".format(
                maturity
            )
        )

    raise NotImplementedError(
        "VolMem training integration is not implemented. Do not route this "
        "configuration through the legacy 2D training entry."
    )


if __name__ == "__main__":
    sys.exit(main())
