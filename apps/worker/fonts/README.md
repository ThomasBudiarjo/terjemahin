# Fonts

`ComicShanns2.ttf` (Comic Shanns by Shannon Miwa, MIT license, in git) is the default
lettering font of the `default` and `basic` renderers (`FONT_PATH`). It covers Latin, so it
fits Indonesian and English targets.

Use another font per run or by default:

```sh
terjemahin-worker translate page.png --opt renderer.font_path=/path/to/Font.ttf
FONT_PATH=/path/to/Font.ttf terjemahin-worker web
```

A missing `FONT_PATH` file makes `default` fail at build time with a clear error; `basic` falls
back to Pillow's built-in font.

## CJK targets (`--to ja|zh|ko`, `-r vertical`)

The Latin font has no kana or hanzi. The vertical renderer (and `default`, which switches to
vertical columns for CJK targets) looks for a CJK font in this order: the
`renderer.cjk_font_path` option (or `font_path` for `-r vertical`), files in this folder
matching `*CJK*`, `*Noto*JP*`, `*gothic*`, `*mincho*` or `*.ttc`, then fontconfig
(`fc-match :lang=ja`). Without one, CJK text renders as boxes and a warning is logged. In the
Docker image, build with `--build-arg CJK_FONTS=1` (installs `fonts-noto-cjk`).

## Other lettering fonts (check the license before commercial use)

- Komika Text / Komika Axis (Apostrophic Labs, free)
- Anime Ace 2.0 BB (Blambot, free for indie use)
- CC Wild Words (Comicraft, paid, the industry standard)
