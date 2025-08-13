"""This script prepares the training dataset for HMESymbolicNet.

The script assumes it's not moved elsewhere i.e (scripts/data_prep/hme.py).

This script:
    1. If there is no dataset/hme/mathwriting-2024 then
        download dataset from https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz.
        Reference: https://arxiv.org/abs/2404.10690.
    2.
"""

import os
from pathlib import Path
from tqdm import tqdm
import requests
import tarfile
import json

import glob
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MultiLabelBinarizer
import re

from collections import Counter, defaultdict
from PIL import Image, ImageDraw

PRINT_BAR_LENGTH = 100  # the length of ==========
IMG_HEIGHT = 128
TOP_COMMON_EQN = 200

#  ===================== helper functions ======================


def download_file(url, out_path):
    # Stream the request so we can download in chunks
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total file size (in bytes) from headers, if available
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # Download in 1 KB chunks

    # Open the file and write chunks with a progress bar
    with (
        open(out_path, "wb") as file,
        tqdm(
            total=total_size,
            unit="B",  # Unit: bytes
            unit_scale=True,  # Scale units automatically
            unit_divisor=1024,  # Use KB/MB instead of 1000-based
            desc=out_path,  # Progress bar label
            initial=0,
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)
                progress_bar.update(len(chunk))


def extract_tgz_bytes_progress(tgz_path: Path, out_dir: Path | None = None):
    tgz_path = Path(tgz_path)
    out_dir = Path(out_dir) if out_dir else tgz_path.parent

    with tarfile.open(tgz_path, "r:gz") as tar:
        members = tar.getmembers()
        total_bytes = sum(m.size for m in members if m.isreg())
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Extracting {tgz_path.name}",
        ) as pbar:
            for m in members:
                tar.extract(m, path=out_dir)
                if m.isreg():
                    pbar.update(m.size)


def generate_vocab(symbol_path: Path, out_path: Path):
    unique_symbols = set()
    with open(symbol_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            unique_symbols.add(obj["label"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(unique_symbols)), f, indent=2)


def generate_multi_hot_label(vocab_path, dataset_path, out_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        symbol_list = json.load(f)

    # gather (filename, tokens) pairs
    filenames = []
    token_lists = []
    ns = {"ink": "http://www.w3.org/2003/InkML"}

    length_sorted_symbol = sorted(symbol_list, key=len)

    for split in ["train", "valid", "test", "symbols"]:
        pattern = os.path.join(dataset_path, split, "*.inkml")
        for inkml_file in tqdm(
            glob.glob(pattern),
            desc=f"Parsing & tokenizing inkML files ({split})",
            unit="file",
            dynamic_ncols=True,
        ):
            # parse InkML, extract the normalizedLabel text
            tree = ET.parse(inkml_file)
            root = tree.getroot()
            ann = root.find(".//ink:annotation[@type='normalizedLabel']", namespaces=ns)
            if ann is None or not ann.text:
                continue  # skip if missing

            expr = ann.text.strip()

            pattern = re.compile("|".join(re.escape(s) for s in length_sorted_symbol))
            tokens = pattern.findall(expr)

            filenames.append(os.path.basename(inkml_file).replace(".inkml", ".png"))
            token_lists.append(tokens)

    print("Writing test config")
    # fit a MultiLabelBinarizer (one-hot for each sample over fixed vocab)
    mlb = MultiLabelBinarizer(classes=symbol_list, sparse_output=False)
    multi_hot_matrix = mlb.fit_transform(token_lists)
    # shape = (n_samples, len(SYMBOL_LIST)), each row sums ≥1

    # write out a JSON mapping filename to one-hot list
    with open(out_path, "w", encoding="utf-8") as f:
        # convert each row to a plain list of 0/1
        vect_data = {
            fname: multi_hot_matrix[i].tolist() for i, fname in enumerate(filenames)
        }
        json.dump(vect_data, f, ensure_ascii=False, indent=2)


def generate_test_files_config(dataset_path, out_path):
    result = []
    eqn_to_filename = defaultdict(list)

    def extract_label(file_path):
        # parse InkML
        tree = ET.parse(file_path)
        root = tree.getroot()
        uri = root.tag.split("}")[0].strip("{")
        ns = {"ink": uri}
        filename = os.path.basename(file_path)

        # extract the ground-truth label
        #    look for <annotation type="truth">...</annotation>
        eqn = None
        for ann in root.findall(".//ink:annotation", ns):
            if ann.get("type") == "truth" or eqn is None:
                eqn = ann.text.strip()
                if ann.get("type") == "truth":
                    break

        eqn_to_filename[eqn].append(filename)
        result.append(eqn)

    for stuff in ["train", "valid", "test", "symbols"]:
        parent = f"{dataset_path}/{stuff}"
        dirs = glob.glob(parent + "/*.inkml")

        for d in tqdm(
            dirs, unit="file", desc=f"Splitting train/test, reading '{stuff}' portion"
        ):
            try:
                extract_label(d)
            except Exception:
                continue

    count = Counter(result)
    most_common = count.most_common(TOP_COMMON_EQN)
    total_data_count = len(result)
    test_data_count = sum([c for _, c in most_common])
    train_data_count = total_data_count - test_data_count

    print(
        f"Train: {train_data_count}, test: {test_data_count}, total: {total_data_count}"
    )

    print("Writing config to differentiate train vs test")
    with open(out_path, "w", encoding="utf-8") as f:
        test_data = {
            file.replace(".inkml", ".png"): list(eqn_to_filename.keys()).index(eq)
            for eq, _ in most_common
            for file in eqn_to_filename[eq]
        }

        json.dump(test_data, f, indent=2)


def render_inkml(dataset_path, test_config_path, out_path):
    # load the filenames that are allocated for test dset
    with open(test_config_path, "r", encoding="utf-8") as f:
        test_filenames = set(json.load(f).keys())

    def render_inkml_with_height(inkml_path, img_height=128, line_width=2, padding=10):
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        # Parse all points from all traces
        traces = []
        for trace in root.findall(".//{http://www.w3.org/2003/InkML}trace"):
            points = []
            for pair in trace.text.strip().split(","):
                pts = pair.strip().split()
                if len(pts) >= 2:
                    x, y = float(pts[0]), float(pts[1])
                    points.append((x, y))
            if points:
                traces.append(points)

        all_points = [pt for stroke in traces for pt in stroke]
        if not all_points:
            raise ValueError("No points found in InkML!")
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute width to preserve aspect ratio
        orig_h = max_y - min_y
        orig_w = max_x - min_x
        scale = (img_height - 2 * padding) / orig_h if orig_h != 0 else 1
        img_width = int(orig_w * scale) + 2 * padding if orig_w != 0 else img_height

        def norm(x, y):
            nx = int((x - min_x) * scale) + padding
            ny = int((y - min_y) * scale) + padding
            return nx, ny

        img = Image.new("L", (img_width, img_height), 255)
        draw = ImageDraw.Draw(img)
        for stroke in traces:
            if len(stroke) > 1:
                draw.line([norm(x, y) for (x, y) in stroke], fill=0, width=line_width)
            elif len(stroke) == 1:
                draw.point(norm(*stroke[0]), fill=0)
        return img

    for name in ["train", "valid", "test", "symbols"]:
        for filepath in tqdm(
            glob.glob(f"{dataset_path}/{name}/*.inkml"),
            desc=f"Rendering '{name}' folder of original data",
            unit="file",
        ):
            try:
                fname = os.path.basename(filepath)
                if fname.replace(".inkml", ".png") in test_filenames:
                    out_subdir = "test"
                else:
                    out_subdir = "train"

                img = render_inkml_with_height(filepath, img_height=IMG_HEIGHT)
                out_dir = f"{out_path}/{out_subdir}"
                os.makedirs(out_dir, exist_ok=True)
                img.save(f"{out_dir}/{fname.replace(".inkml", ".png")}")
            except ET.ParseError:
                print("Parse error: ", filepath)
                continue


def main():
    # get project root
    root = Path(__file__).parent.parent.parent
    dataset_path = root / "dataset/hme"

    # create dataset/hme if doesn't exist
    os.makedirs(root / dataset_path, exist_ok=True)

    # check if dataset/hme/mathwriting-2024 exist
    target_path = dataset_path / "mathwriting-2024.tgz"

    # 1. Download dataset if it's not found
    print("1. Starting to download dataset")

    if os.path.exists(target_path):
        relative_path = target_path.relative_to(root)
        print(
            f"Found dataset, ensure dataset was successfully downloaded (2.88 GB), otherwise remove {relative_path}"
        )

    else:
        download_file(
            url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz",
            out_path=str(target_path.parent / "mathwriting-2024.tgz"),
        )
        print("Dataset has been downloaded")

    # 2. unzip tgz
    print(f"2. Unzipping {target_path}")
    extract_tgz_bytes_progress(target_path, dataset_path)

    unzipped_data_path = dataset_path / "mathwriting-2024"
    generated_data_path = dataset_path / "generated_data"
    vocab_path = generated_data_path / "vocab.json"
    multi_hot_path = generated_data_path / "multihot_label.json"
    test_config_path = generated_data_path / "test.json"
    rendered_path = generated_data_path / "image"

    # generate dirs for storing dynamically generated infos eg: vocab
    os.makedirs(generated_data_path, exist_ok=True)

    # 3. Generate vocabulary
    print("3. Generating vocabulary for HME")
    generate_vocab(
        symbol_path=unzipped_data_path / "symbols.jsonl", out_path=vocab_path
    )

    # 4. Build one hot vector for eack inkml
    print("4. Building one hot vector for eack inkml")
    generate_multi_hot_label(vocab_path, unzipped_data_path, multi_hot_path)

    # 5. Make most common equations into test while others as train
    print("5. Splitting training and testing dataset")
    generate_test_files_config(unzipped_data_path, test_config_path)

    # 6. Render inkml
    print("6. Rendering all inkml to PNGs")
    render_inkml(unzipped_data_path, test_config_path, rendered_path)


if __name__ == "__main__":
    main()
