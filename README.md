# WorldReasonBench &mdash; Project Page

This branch (`gh-pages`) hosts the project page at
[https://unix-ai-lab.github.io/WorldReasonBench/](https://unix-ai-lab.github.io/WorldReasonBench/).

For the **code, data, and evaluation pipelines**, switch to the
[`main`](https://github.com/UniX-AI-Lab/WorldReasonBench/tree/main) branch.

## Local preview

```bash
git checkout gh-pages
python3 -m http.server 8000
# open http://localhost:8000
```

## Layout

```
.
├── index.html                     # entry point
├── .nojekyll                      # disable Jekyll build, serve files as-is
├── data/
│   └── leaderboard.json           # main-table data (Score_PR + S(v))
└── static/
    ├── css/style.css
    ├── js/main.js                 # leaderboard, video tabs, counters
    ├── images/                    # PNGs converted from paper figures
    └── videos/                    # qualitative MP4 examples (placeholder)
```

## Updating the leaderboard

Edit `data/leaderboard.json`. The page reloads numbers without any rebuild.

## Adding qualitative videos

Drop MP4 files into `static/videos/` and update the `videoData` map in
`static/js/main.js`. See `static/videos/README.md` for the recommended ffmpeg
encoding command and naming convention.
