"""
Core logic for the S-data / I-data personality agreement exhibit.

100 trait-descriptive adjectives, organized the way personality research
organizes them: 5 Big Five dimensions x 2 poles (10 "high" + 10 "low"
markers each) = 100 words total. This structure is inspired by Lewis
Goldberg's well-known "100 unipolar Big-Five markers" (Goldberg, 1992,
Psychological Assessment) -- a real, widely-cited personality assessment
instrument -- though the exact word list here has been assembled from
commonly-used Big Five marker terms rather than reproduced verbatim from
the original copyrighted instrument.

Each adjective maps to (trait, sign). When a person picks 10 words, we
sum the signed hits per trait, giving a 5-number personality *profile*
instead of one flattened number. Two profiles (self vs. informant) are
then compared with a Pearson correlation -- an actual "correlation
analysis" across 5 real psychological dimensions, rather than an
arbitrary single value line.
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; Streamlit sets its own at runtime
import matplotlib.pyplot as plt

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

ADJECTIVES = {
    # --- Openness ---
    "Creative": ("Openness", 1),
    "Imaginative": ("Openness", 1),
    "Curious": ("Openness", 1),
    "Philosophical": ("Openness", 1),
    "Original": ("Openness", 1),
    "Artistic": ("Openness", 1),
    "Insightful": ("Openness", 1),
    "Inventive": ("Openness", 1),
    "Reflective": ("Openness", 1),
    "Complex": ("Openness", 1),

    "Grounded": ("Openness", -1),
    "Conventional": ("Openness", -1),
    "Straightforward": ("Openness", -1),
    "Traditional": ("Openness", -1),
    "Single-minded": ("Openness", -1),
    "Unknowing": ("Openness", -1),
    "Simple": ("Openness", -1),
    "Predictable": ("Openness", -1),
    "Surface-level": ("Openness", -1),
    "Frivolous": ("Openness", -1),

    # --- Conscientiousness ---
    "Organized": ("Conscientiousness", 1),
    "Efficient": ("Conscientiousness", 1),
    "Thorough": ("Conscientiousness", 1),
    "Systematic": ("Conscientiousness", 1),
    "Reliable": ("Conscientiousness", 1),
    "Disciplined": ("Conscientiousness", 1),
    "Careful": ("Conscientiousness", 1),
    "Precise": ("Conscientiousness", 1),
    "Punctual": ("Conscientiousness", 1),
    "Practical": ("Conscientiousness", 1),

    "Flexible": ("Conscientiousness", -1),
    "Laid-back": ("Conscientiousness", -1),
    "Unhurried": ("Conscientiousness", -1),
    "Casual": ("Conscientiousness", -1),
    "Unplanned": ("Conscientiousness", -1),
    "Inconsistent": ("Conscientiousness", -1),
    "Informal": ("Conscientiousness", -1),
    "Unconventional": ("Conscientiousness", -1),
    "Adaptable": ("Conscientiousness", -1),
    "Absent-minded": ("Conscientiousness", -1),

    # --- Extraversion ---
    "Talkative": ("Extraversion", 1),
    "Outgoing": ("Extraversion", 1),
    "Energetic": ("Extraversion", 1),
    "Bold": ("Extraversion", 1),
    "Assertive": ("Extraversion", 1),
    "Lively": ("Extraversion", 1),
    "Sociable": ("Extraversion", 1),
    "Adventurous": ("Extraversion", 1),
    "Enthusiastic": ("Extraversion", 1),
    "Spontaneous": ("Extraversion", 1),

    "Quiet": ("Extraversion", -1),
    "Shy": ("Extraversion", -1),
    "Bashful": ("Extraversion", -1),
    "Timid": ("Extraversion", -1),
    "Introverted": ("Extraversion", -1),
    "Withdrawn": ("Extraversion", -1),
    "Reserved": ("Extraversion", -1),
    "Unadventurous": ("Extraversion", -1),
    "Passive": ("Extraversion", -1),
    "Reticent": ("Extraversion", -1),

    # --- Agreeableness ---
    "Kind": ("Agreeableness", 1),
    "Warm": ("Agreeableness", 1),
    "Sympathetic": ("Agreeableness", 1),
    "Cooperative": ("Agreeableness", 1),
    "Generous": ("Agreeableness", 1),
    "Courteous": ("Agreeableness", 1),
    "Trusting": ("Agreeableness", 1),
    "Considerate": ("Agreeableness", 1),
    "Gentle": ("Agreeableness", 1),
    "Compassionate": ("Agreeableness", 1),

    "Distant": ("Agreeableness", -1),
    "Blunt": ("Agreeableness", -1),
    "Direct": ("Agreeableness", -1),
    "Private": ("Agreeableness", -1),
    "Independent": ("Agreeableness", -1),
    "Self-focused": ("Agreeableness", -1),
    "Strong-willed": ("Agreeableness", -1),
    "Cautious": ("Agreeableness", -1),
    "Matter-of-fact": ("Agreeableness", -1),
    "Self-assured": ("Agreeableness", -1),

    # --- Neuroticism ---
    "Anxious": ("Neuroticism", 1),
    "Moody": ("Neuroticism", 1),
    "Nervous": ("Neuroticism", 1),
    "Touchy": ("Neuroticism", 1),
    "Self-conscious": ("Neuroticism", 1),
    "Temperamental": ("Neuroticism", 1),
    "Worried": ("Neuroticism", 1),
    "Tense": ("Neuroticism", 1),
    "Competitive": ("Neuroticism", 1),
    "Fretful": ("Neuroticism", 1),

    "Calm": ("Neuroticism", -1),
    "Relaxed": ("Neuroticism", -1),
    "Secure": ("Neuroticism", -1),
    "Contented": ("Neuroticism", -1),
    "Stable": ("Neuroticism", -1),
    "Easygoing": ("Neuroticism", -1),
    "Unworried": ("Neuroticism", -1),
    "Composed": ("Neuroticism", -1),
    "Steady": ("Neuroticism", -1),
    "Even-tempered": ("Neuroticism", -1),
}
assert len(ADJECTIVES) == 100, f"Expected 100 adjectives, got {len(ADJECTIVES)}"
for _t in TRAITS:
    _pos = sum(1 for v, s in ADJECTIVES.values() if v == _t and s == 1)
    _neg = sum(1 for v, s in ADJECTIVES.values() if v == _t and s == -1)
    assert _pos == 10 and _neg == 10, f"{_t}: {_pos} positive / {_neg} negative (expected 10/10)"


def compute_profile(words):
    """Turn a list of 10 chosen adjectives into a 5-number Big Five profile."""
    profile = {t: 0 for t in TRAITS}
    for w in words:
        trait, sign = ADJECTIVES[w]
        profile[trait] += sign
    return np.array([profile[t] for t in TRAITS], dtype=float)


def compute_score(self_words, informant_words):
    """
    Compare a self-report word set (S-data) against an informant word set
    (I-data) and return:
      score     -- 1-10 scale, higher = the informant's picture of the
                   person lines up more closely with the person's own
      r         -- the underlying Pearson correlation between the two
                   5-dimension trait profiles (None if undefined, i.e.
                   one profile has zero variance across all 5 traits)
      overlap   -- how many of the exact same words both people picked (0-10)
      v1, v2    -- the raw 5-number profiles (self, informant) for plotting
    """
    v1 = compute_profile(self_words)
    v2 = compute_profile(informant_words)
    overlap = len(set(self_words) & set(informant_words))

    if np.std(v1) == 0 or np.std(v2) == 0:
        # Profile correlation is undefined when one profile is perfectly flat
        # (can happen if picks cancel out evenly across all 5 traits).
        # Fall back to scaling directly off word overlap instead.
        r = None
        score = round((overlap / 10) * 9 + 1, 1)
    else:
        r, _ = pearsonr(v1, v2)
        r = float(np.clip(r, -1, 1))
        score = round(((r + 1) / 2) * 9 + 1, 1)

    return score, r, overlap, v1, v2


# A person picks exactly 10 words, so in the extreme case all 10 land on one
# trait with the same sign -- a single trait score can range from -10 to +10.
MAX_TRAIT_SCORE = 10


def _radius(v, lo=-MAX_TRAIT_SCORE, hi=MAX_TRAIT_SCORE, r_lo=0.2, r_hi=2.2):
    """
    Map a signed trait score onto a radius that never collapses opposite-sign
    scores onto the same ring.

    A naive radar chart plots radius = abs(value) (or clips negatives to 0),
    since polar radius is conventionally non-negative. That throws away the
    sign: a trait scored +10 and one scored -10 both land on the same ring,
    so a pair of PERFECTLY INVERSE profiles (r = -1, the worst possible
    agreement) can render as a single overlapping shape -- the exact
    opposite of what the chart is supposed to show.

    Shifting the whole [-MAX_TRAIT_SCORE, +MAX_TRAIT_SCORE] range onto
    [r_lo, r_hi] instead keeps both sign and magnitude visible: -10 sits
    near the center, 0 sits mid-ring, +10 sits on the outer edge.
    """
    return r_lo + (v - lo) / (hi - lo) * (r_hi - r_lo)


def plot_radar(v1, v2, score=None, r=None, title=None):
    """
    Sign-preserving radar chart comparing the self (v1) and informant (v2)
    Big Five profiles returned by compute_score().

    Pass `score` / `r` straight from compute_score() to auto-build a title
    (handles the word-overlap fallback case where r is None), or pass your
    own `title` string directly. Returns a matplotlib Figure -- use
    fig.savefig(...) standalone, or st.pyplot(fig) inside the Streamlit app.
    """
    if title is None:
        if score is not None and r is not None:
            title = f"Agreement score: {score}/10 (r={r:.2f})"
        elif score is not None:
            title = f"Agreement score: {score}/10 (based on word overlap)"
        else:
            title = "S-data vs I-data profile"

    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist()
    angles += angles[:1]

    r1 = [_radius(v) for v in v1] + [_radius(v1[0])]
    r2 = [_radius(v) for v in v2] + [_radius(v2[0])]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
    ax.plot(angles, r1, "o-", label="Self (S-data)")
    ax.fill(angles, r1, alpha=0.15)
    ax.plot(angles, r2, "o-", label="Informant (I-data)")
    ax.fill(angles, r2, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TRAITS)
    ax.set_yticklabels([])  # shifted radii aren't meaningful as tick labels
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig
