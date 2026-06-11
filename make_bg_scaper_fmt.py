"""
make_bg_scaper_fmt.py

FSD50K dev.csv를 읽어 bg_scaper_fmt/train(valid)/[클래스명]/ 구조를 생성.
add_interference.py가 이 구조를 기대함.

사용법:
    python make_bg_scaper_fmt.py \
        --fsd_dir data/FSD50K \
        --output_dir data/bg_scaper_fmt
"""

import os
import argparse
import pandas as pd
from pathlib import Path

SELECTED_CLASSES = [
    "Air conditioning", "Aircraft", "Bird flight, flapping wings", "Bleat", "Boiling", "Boom",
    "Burping, eructation", "Burst, pop", "Bus", "Camera", "Car passing by",
    "Cattle, bovinae", "Chainsaw", "Chewing, mastication", "Chink, clink", "Clip-clop",
    "Cluck", "Clunk", "Coin (dropping)", "Crack", "Crackle", "Creak", "Croak", "Crow",
    "Crumpling, crinkling", "Crushing", "Drill", "Drip", "Electric toothbrush", "Engine",
    "Fart", "Finger snapping", "Fire", "Fire alarm", "Firecracker", "Fireworks",
    "Fixed-wing aircraft, airplane", "Frog", "Gears", "Growling", "Gurgling", "Helicopter",
    "Hiss", "Hoot", "Howl", "Howl (wind)", "Jackhammer", "Keys jangling", "Lawn mower",
    "Light engine (high frequency)", "Microwave oven", "Moo", "Oink", "Packing tape, duct tape",
    "Pig", "Printer", "Purr", "Rain", "Rain on surface", "Raindrop", "Ratchet, pawl",
    "Rattle", "Sanding", "Sawing", "Scissors", "Screech", "Sheep", "Ship",
    "Shuffling cards", "Skateboard", "Slam", "Sliding door", "Sneeze", "Sniff", "Snoring",
    "Splinter", "Squeak", "Stream", "Subway, metro, underground", "Tap", "Tearing",
    "Thump, thud", "Tick", "Tick-tock", "Toothbrush", "Traffic noise, roadway noise",
    "Train", "Train horn", "Velcro, hook and loop fastener", "Waterfall",
    "Whoosh, swoosh, swish", "Wind", "Writing", "Zipper (clothing)"
]

SPLIT_MAP = {"train": "train", "val": "valid"}


def normalize(label: str) -> str:
    return label.replace("_", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd_dir", type=str, default="data/FSD50K",
                        help="FSD50K 루트 (FSD50K.dev_audio, FSD50K.ground_truth 포함)")
    parser.add_argument("--output_dir", type=str, default="data/bg_scaper_fmt",
                        help="bg_scaper_fmt 출력 경로")
    args = parser.parse_args()

    fsd_dir   = Path(args.fsd_dir)
    bg_dir    = Path(args.output_dir)
    dev_csv   = fsd_dir / "FSD50K.ground_truth/dev.csv"
    audio_dir = fsd_dir / "FSD50K.dev_audio/FSD50K.dev_audio"

    assert dev_csv.exists(),   f"dev.csv 없음: {dev_csv}"
    assert audio_dir.exists(), f"dev_audio 없음: {audio_dir}"

    selected_set = set(SELECTED_CLASSES)

    # 1) 빈 폴더 먼저 생성 (없으면 add_interference.py assert 실패)
    print("빈 폴더 생성 중...")
    for split in ("train", "valid"):
        for cls in SELECTED_CLASSES:
            (bg_dir / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"  train/valid × {len(SELECTED_CLASSES)}클래스 폴더 생성 완료")

    # 2) dev.csv 읽어 심링크 생성
    print("심링크 생성 중...")
    df = pd.read_csv(dev_csv)
    copied = skipped_no_wav = skipped_no_class = 0

    for _, row in df.iterrows():
        fname = str(row["fname"])
        split = SPLIT_MAP.get(row["split"])
        if split is None:
            continue

        wav_src = audio_dir / f"{fname}.wav"
        if not wav_src.exists():
            skipped_no_wav += 1
            continue

        labels  = [normalize(l) for l in str(row["labels"]).split(",")]
        matched = [l for l in labels if l in selected_set]
        if not matched:
            skipped_no_class += 1
            continue

        for label in matched:
            dest = bg_dir / split / label / f"{fname}.wav"
            if not dest.exists():
                os.symlink(wav_src.resolve(), dest)
                copied += 1

    print(f"  심링크 생성:         {copied}")
    print(f"  wav 없음 skip:       {skipped_no_wav}")
    print(f"  클래스 미해당 skip:  {skipped_no_class}")

    # 3) 커버 현황
    covered = set()
    for split in ("train", "valid"):
        for p in (bg_dir / split).iterdir():
            if p.is_dir() and any(p.iterdir()):
                covered.add(p.name)

    missing = selected_set - covered
    print(f"\n커버된 클래스: {len(covered)}/94")
    if missing:
        print(f"FSD50K에 없는 클래스 ({len(missing)}개):")
        for c in sorted(missing):
            print(f"  - {c}")
    print("\n완료. add_interference.py 실행 가능.")


if __name__ == "__main__":
    main()
