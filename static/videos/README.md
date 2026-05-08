# Video assets

Place qualitative example videos here, one MP4 per file.

Suggested naming convention:

```
wk_<short-id>.mp4    # World Knowledge
hc_<short-id>.mp4    # Human-Centric
lr_<short-id>.mp4    # Logic Reasoning
ib_<short-id>.mp4    # Information-Based
```

Recommended encoding for the project page (small file size, broad browser support):

```bash
ffmpeg -i input.mp4 \
  -vcodec libx264 -crf 26 -preset slow -profile:v main -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  -vf "scale='min(960,iw)':-2" \
  output.mp4
```

After the videos are added, update the `videoData` map in `static/js/main.js` to
point at real files via the `src` field, e.g.:

```js
{ title: '...', cat: 'World Knowledge', src: 'static/videos/wk_apple.mp4' }
```

and adjust `renderVideos()` to render `<video controls>` instead of the
placeholder poster card.
