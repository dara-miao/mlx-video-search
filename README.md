# mlx-video-search

Natural-language search over local video, running on [MLX](https://github.com/ml-explore/mlx) — Apple’s array framework for Apple Silicon.

It samples frames, captions them with a vision-language model, then matches a query like “the moment we hit the water” to a clip and timestamp. Nothing leaves the machine.

The default model is [Qwen3-VL 4B (4-bit)](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit), loaded through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Pass `--model` to use another mlx-vlm id.

## Requirements

- Apple Silicon Mac
- Python 3.10+

The first index downloads those Qwen3-VL weights from Hugging Face (~3 GB).

## Install

```bash
git clone https://github.com/dara-miao/mlx-video-search.git
cd mlx-video-search
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## App

```bash
mlx-video-search
```

Opens a local UI. Choose a folder, index it, search. Hits include a frame, timecode, and buttons to open the file in QuickTime or Finder.

## CLI

Index a folder of clips:

```bash
mlx-video-search ~/Desktop/broll
```

Search the index:

```bash
mlx-video-search ~/Desktop/broll "the moment we hit the water"
```

The index is `mlx-video-index.json` in that folder. Later runs skip videos already in it.

```bash
mlx-video-search --help
```

## How it works

1. Sample frames from each video (default: one per second).
2. Run the VLM on each frame and store a caption plus structured tags.
3. On search, interpret the query, retrieve candidate frames, then confirm against the pixels.

## License

Not set yet.
