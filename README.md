# mlx-video-search

Natural-language search over local video, running on [MLX](https://github.com/ml-explore/mlx).

It samples frames, captions them with a vision-language model, then matches a query like “the moment I look away” to a clip and timestamp. Local on your computer.

The default model is [Qwen3-VL 4B (4-bit)](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit), loaded through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Pass `--model` to use another mlx-vlm id.

MLX is built for Apple chips, so this runs on a Mac with Apple Silicon and Python 3.10+. The first index downloads those Qwen3-VL weights from Hugging Face (~3 GB).

## Install

```bash
git clone https://github.com/dara-miao/mlx-video-search.git
cd mlx-video-search
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mkdir -p ~/.local/bin
ln -sf "$(pwd)/.venv/bin/mlx-video-search" ~/.local/bin/mlx-video-search
```

## App

```bash
mlx-video-search
```

Opens a local UI. Choose a folder, index it, search. Hits include a frame, timecode, and buttons to open the file in QuickTime or Finder.

You can ask for a glance, a sport phase (`golf swing at impact`, `backswing`), a place (`Marbella`, `kitchen`), or a short category (`jump`, `sport`). iPhone clips often have GPS; search reads that on first look and can match the course or town, not only what’s in the caption.

## CLI

Index a folder of clips:

```bash
mlx-video-search ~/Desktop/broll
```

Search the index:

```bash
mlx-video-search ~/Desktop/broll "the dog running in"
```

The index is `mlx-video-index.json` in that folder. Later runs skip videos already in it. Re-index after a model or prompt change if you want new caption fields (gaze, sport phase). GPS place names attach without re-indexing.

```bash
mlx-video-search --help
```

## How it works

1. Sample frames from each video (default: one per second).
2. Run the VLM on each frame and store a caption plus tags (scene, gaze, sport phase).
3. Read GPS from the file when it’s there, and reverse-geocode it to a place name.
4. On search, keep the query as written. Use captions and places to pick frames, then confirm against the pixels. Hits show up while it is still looking.

## License

[MIT](LICENSE)
