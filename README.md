# Spotify streaming history

Explore your full Spotify listening history: what you listened to, when, and
how it changed over the years.

Upload the zip file Spotify sends you, and the app shows your top songs,
artists and albums, podcasts and audiobooks, your listening habits by hour and
weekday, and your own "Wrapped" for every year.

## Your data stays on your computer

The web app runs entirely in your browser with
[stlite](https://github.com/whitphx/stlite): Python and
[Streamlit](https://streamlit.io) run inside the browser (via
[Pyodide](https://pyodide.org)). The page is a set of static files; there is
no server that could receive your data. Your zip file is read in your browser
and never uploaded anywhere.

## Get your data

1. Go to your Spotify account's
   [privacy settings](https://www.spotify.com/account/privacy/).
2. Request your **Extended streaming history**. It covers your whole account
   history, unlike the "Account data", which only covers the last year.
3. After Spotify emails you (this can take up to 30 days), download the zip
   file.

Upload the zip as it is; there is no need to unzip it.

## What you see

- **Overview**: key figures, music vs. podcasts vs. audiobooks, skips and
  shuffle, devices, countries, play lengths, and when you listen (hour ×
  weekday).
- **Music**: top songs, artists and albums, the most skipped ones, and a
  detail view for a single artist, song or album.
- **Podcasts** and **Audiobooks** (only if you listened to any): top shows,
  episodes and audiobooks, and listening over time.
- **Over the years**: listening per month, your top artists and songs of each
  year, how the all-time top 10 developed, new artists per year, a calendar
  heatmap, and your record days.

Everything can be limited to a period and counted in plays or in hours.

## Run it yourself

You need [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repository>
cd spotify-analysis
uv sync
uv run streamlit run app/streamlit_app.py
```

To try the browser version ([app/index.html](app/index.html)) locally, serve
the project folder and open http://localhost:8000/app/:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

## Project structure

```
src/spotify_analysis/
  load.py         read the zip and clean the data (local time, platforms, …)
  analysis.py     calculations: top lists, listening over time, rankings, …
  charts.py       Plotly figures
app/
  streamlit_app.py  the Streamlit app
  index.html        the browser version: loads the app with stlite
```

`load`, `analysis` and `charts` don't depend on Streamlit, so they can also be
used in a notebook.

## Notes

- **All plays count**, however short. To count only plays of at least 30
  seconds, like Spotify does, set `MIN_PLAY_MS` in
  [analysis.py](src/spotify_analysis/analysis.py).
- **Songs are identified by artist and song name**, not by Spotify's ID: the
  same song often has several IDs (album version, single, remaster), which
  would split its plays.
- **Local time** is derived from the country of each play. For countries with
  several time zones, the main one is used (e.g. Madrid for all of Spain,
  including the Canary Islands).
