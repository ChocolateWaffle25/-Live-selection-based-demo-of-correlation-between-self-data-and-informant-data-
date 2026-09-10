import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scoring import ADJECTIVES, TRAITS, compute_score

st.set_page_config(page_title="How Well Do You Know Me?", page_icon="\U0001F9E9", layout="centered")

WORD_LIST = sorted(ADJECTIVES.keys())
PICK_N = 10


def init_state():
    defaults = {
        "stage": "intro",
        "self_words": [],
        "informant_words": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset():
    st.session_state.stage = "intro"
    st.session_state.self_words = []
    st.session_state.informant_words = []


def word_picker(key, heading, caption, next_label):
    st.subheader(heading)
    st.caption(caption)
    selected = st.multiselect(
        f"Pick exactly {PICK_N} words:",
        WORD_LIST,
        default=st.session_state[key],
        key=f"{key}_widget",
    )
    st.write(f"Selected: **{len(selected)} / {PICK_N}**")
    if len(selected) > PICK_N:
        st.warning(f"That's {len(selected)} words — please remove {len(selected) - PICK_N} to continue.")
    ready = len(selected) == PICK_N
    if st.button(next_label, type="primary", disabled=not ready, use_container_width=True):
        st.session_state[key] = selected
        return True
    return False


def radar_chart(v1, v2):
    n = len(TRAITS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_plot = angles + angles[:1]

    # Each trait score can range roughly -10..+10 (only 10 words are ever
    # picked, so a trait can't accumulate more than +/-10). Polar plots
    # reflect negative radii to the opposite side of the circle, which
    # distorts the shape -- so shift everything up by 10 to keep radius
    # non-negative. set_yticklabels([]) below hides the now-meaningless
    # raw axis numbers; only the shape/overlap between the two profiles
    # matters, not the absolute numbers.
    OFFSET = 10
    v1_plot = [x + OFFSET for x in v1] + [v1[0] + OFFSET]
    v2_plot = [x + OFFSET for x in v2] + [v2[0] + OFFSET]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)   # first trait starts at 12 o'clock
    ax.set_theta_direction(-1)       # go clockwise

    line1, = ax.plot(angles_plot, v1_plot, linewidth=2, color="#4C72B0", label="Self (S-data)")
    ax.fill(angles_plot, v1_plot, alpha=0.15, color="#4C72B0")
    line2, = ax.plot(angles_plot, v2_plot, linewidth=2, color="#DD8452", label="Informant (I-data)")
    ax.fill(angles_plot, v2_plot, alpha=0.15, color="#DD8452")

    ax.set_xticks(angles)
    ax.set_xticklabels(TRAITS, fontsize=10)
    ax.tick_params(axis="x", pad=22)  # push trait names outward, clear of the plot
    ax.set_ylim(0, 2 * OFFSET)
    ax.set_yticklabels([])

    # Reserve space below the chart so the legend has clear room of its own,
    # instead of floating over the plot or the trait labels.
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.14, right=0.86)
    fig.legend(handles=[line1, line2], loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02))

    return fig


init_state()
st.title("\U0001F9E9 How Well Do You Know Me?")
st.caption("A live demo of self-report vs. informant-report agreement in personality psychology")

if st.session_state.stage == "intro":
    st.write(
        "In personality psychology, **S-data** is how you describe yourself, and **I-data** "
        "is how someone who knows you describes you. Researchers compare the two to study "
        "how well people really know each other.\n\n"
        f"Here's how it works:\n"
        f"1. Person 1 picks {PICK_N} words that describe *themselves*.\n"
        f"2. Person 2 (a friend) picks {PICK_N} words that describe *Person 1*.\n"
        "3. We correlate the two personality profiles and turn it into a score out of 10."
    )
    if st.button("Start", type="primary", use_container_width=True):
        st.session_state.stage = "self"
        st.rerun()

elif st.session_state.stage == "self":
    if word_picker(
        "self_words",
        "Step 1 \u2014 Describe yourself",
        "Person 1: choose the words that describe you.",
        "Next",
    ):
        st.session_state.stage = "handoff"
        st.rerun()

elif st.session_state.stage == "handoff":
    st.subheader("\U0001F504 Switch places")
    st.write(
        "Person 1's answers are now hidden. Hand the device to the friend "
        "who's going to describe Person 1."
    )
    if st.button("I'm the friend \u2014 continue", type="primary", use_container_width=True):
        st.session_state.stage = "informant"
        st.rerun()

elif st.session_state.stage == "informant":
    if word_picker(
        "informant_words",
        "Step 2 \u2014 Describe your friend",
        "Person 2: choose the words that describe Person 1.",
        "See results",
    ):
        st.session_state.stage = "results"
        st.rerun()

elif st.session_state.stage == "results":
    score, r, overlap, v1, v2 = compute_score(
        st.session_state.self_words, st.session_state.informant_words
    )
    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("How well do they know you?", f"{score} / 10")
    col2.metric("Exact word overlap", f"{overlap} / {PICK_N}")
    if r is not None:
        st.caption(f"Underlying profile correlation: r = {r:.2f}")
    else:
        st.caption("Profile correlation was undefined for this pair, so the score is based on word overlap.")

    st.pyplot(radar_chart(v1, v2))

    with st.expander("See the words each person picked"):
        st.write("**Self:** " + ", ".join(st.session_state.self_words))
        st.write("**Friend's view:** " + ", ".join(st.session_state.informant_words))

    st.divider()
    if st.button("Run again with a new pair", use_container_width=True):
        reset()
        st.rerun()
