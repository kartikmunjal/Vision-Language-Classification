#!/usr/bin/env python3
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-name", default="Kartik Munjal")
    parser.add_argument("--allowed-email", default="kartikmunjal19@gmail.com")
    args = parser.parse_args()
    raw = subprocess.check_output(["git", "log", "--format=%an%x00%ae"], text=True)
    identities = {tuple(line.split("\0")) for line in raw.splitlines() if line}
    unexpected = identities - {(args.allowed_name, args.allowed_email)}
    if unexpected:
        raise SystemExit(f"unexpected Git author identities: {sorted(unexpected)}")
    print(f"authorship audit passed: {len(identities)} identity")


if __name__ == "__main__":
    main()
