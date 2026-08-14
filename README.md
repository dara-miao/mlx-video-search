# mlx-video-search

Search your own videos in natural language, locally on Apple Silicon.

## App

The usual way to use it: run this once, then work in the browser.

```bash
source ~/Projects/mlx-video-search/.venv/bin/activate
mlx-video-search
```

That opens a local page. Choose a folder, index the library, then type a moment
like “the moment we hit the water”. Results show the clip, timestamp, and a
frame. You can open the shot in QuickTime or reveal it in Finder.

First index run downloads the 4-bit Qwen3-VL weights from Hugging Face (~3 GB).

## Terminal

The CLI is still there if you want it.

```bash
mlx-video-search ~/Desktop/broll
mlx-video-search ~/Desktop/broll "the moment we hit the water"
```
