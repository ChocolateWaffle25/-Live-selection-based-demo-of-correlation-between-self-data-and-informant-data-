# How Well Do You Know Me? — setup

## One-time setup
```
pip install -r requirements.txt
```

## Run it
```
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`. Leave the
terminal window open while the app is running — closing it shuts the app down.

## Files
- `app.py` — the interface (this is what you run)
- `scoring.py` — the 100-word adjective database + the scoring math (no
  Streamlit dependency, so you can test/tweak it on its own — see the
  bottom of this file for a quick sanity check)
- `requirements.txt` — the 4 packages needed

## Sanity-checking the scoring logic on its own
```
python3 -c "from scoring import compute_score; print(compute_score(['Kind','Warm'], ['Kind','Warm']))"
```

## Tweaking things later
- To change the 1-10 scale mapping, edit the `score = ...` line in
  `compute_score()` in `scoring.py`.
- To add/remove/reword adjectives, edit the `ADJECTIVES` dict in
  `scoring.py` — just keep each trait at 10 positive + 10 negative words
  (there's an `assert` at the bottom of the file that will warn you if the
  count gets thrown off).
