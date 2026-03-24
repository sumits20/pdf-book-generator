import streamlit as st
from together import Together
from openai import OpenAI
from fpdf import FPDF
import requests
from io import BytesIO
import gc
import json
import re

# =========================================================
# Session State
# =========================================================

if "page_prompts" not in st.session_state:
    st.session_state.page_prompts = []

if "prompt_generation_done" not in st.session_state:
    st.session_state.prompt_generation_done = False

# =========================================================
# Prompt / Style Settings
# =========================================================

STYLE_SUFFIX = (
    "black and white kids coloring book page, "
    "one subject only, centered on page, "
    "full body visible, simple pose, "
    "thick bold clean black outlines, uniform line thickness, "
    "minimal interior detail, large open coloring spaces, "
    "plain white background, no scenery, no background objects, "
    "no extra characters, rectangular border frame"
)

NEGATIVE_PROMPT = (
    "color, coloured, grayscale, grey fill, gray fill, shading, gradients, shadows, lighting, "
    "dark background, black background, scenery, landscape, sky, clouds, mountains, trees, grass, "
    "multiple characters, crowd, extra animals, toys, party items, balloons, cake, decorations, "
    "3D, realistic, textures, sketch, messy lines, thin lines, blur, noise, "
    "cross-hatching, hatching, text, watermark"
)

# =========================================================
# Streamlit Page
# =========================================================

st.set_page_config(page_title="Together AI KDP Generator", page_icon="📚")
st.title("AI Powered KDP Book Maker")
st.markdown("Generate coloring-book prompts with OpenAI, review them, then create pages with Together AI.")

# =========================================================
# API Clients
# =========================================================

try:
    together_client = Together(api_key=st.secrets["TOGETHER_API_KEY"])
except Exception:
    st.error("Missing TOGETHER_API_KEY in Streamlit Secrets.")
    st.stop()

try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("Book Settings")

    book_title = st.text_input("Book Filename", "my_together_book")
    page_count = st.number_input("Number of Pages", min_value=1, max_value=100, value=10)

    st.divider()
    st.subheader("Theme Input")
    master_prompt = st.text_area(
        "Book Theme",
        "Baby Animals"
    )

    st.divider()
    st.subheader("Model Settings")
    together_model = st.text_input(
        "Together Model",
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    image_steps = st.slider("Image Steps", min_value=20, max_value=60, value=30)
    image_width = st.number_input("Image Width", min_value=512, max_value=1024, value=800, step=64)
    image_height = st.number_input("Image Height", min_value=512, max_value=1408, value=1024, step=64)

# =========================================================
# Helpers
# =========================================================

def extract_json_array(text: str):
    """
    Try to parse a JSON array from model output.
    """
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return None


def build_page_prompt_list(theme: str, count: int) -> list[str]:
    """
    Uses OpenAI GPT-4o mini to generate a clean list of page prompts.
    """

    instructions = (
        "You create short prompt items for a children's coloring book. "
        "Return ONLY a valid JSON array of strings. "
        "Each item must be short, concrete, and suitable for a single coloring page. "
        "Do not include numbering. Do not include explanations. "
        "Keep each prompt focused on one subject only."
    )

    user_input = f"""
Theme: {theme}
Number of pages: {count}

Create exactly {count} short coloring-book page prompts.

Rules:
- Each prompt should describe one single subject only.
- Keep prompts simple and child-friendly.
- Good examples:
  "baby elephant"
  "baby tiger sitting"
  "baby bunny holding one carrot"
- Avoid backgrounds, scenery, multiple subjects, story scenes, and complex compositions.
- Return ONLY JSON array.
"""

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=user_input,
    )

    raw_text = response.output_text.strip()
    prompt_list = extract_json_array(raw_text)

    if not prompt_list:
        raise ValueError(f"Could not parse prompt list from OpenAI response:\n{raw_text}")

    cleaned = []
    for item in prompt_list:
        if isinstance(item, str):
            item = item.strip()
            if item:
                cleaned.append(item)

    if len(cleaned) != count:
        raise ValueError(
            f"Expected {count} prompts, but got {len(cleaned)}.\nParsed prompts: {cleaned}"
        )

    return cleaned


def reset_prompt_inputs(count: int):
    """
    Remove old text_input keys if page count/theme changes and user regenerates prompts.
    """
    for i in range(100):
        key = f"prompt_{i}"
        if key in st.session_state:
            del st.session_state[key]


# =========================================================
# Step 1: Generate Prompt List
# =========================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Prompt List", use_container_width=True):
        try:
            reset_prompt_inputs(int(page_count))

            with st.spinner("Creating page prompt list with OpenAI..."):
                prompts = build_page_prompt_list(master_prompt, int(page_count))

            st.session_state.page_prompts = prompts
            st.session_state.prompt_generation_done = True
            st.success("Prompt list created successfully.")

        except Exception as e:
            st.error(f"Failed to generate prompt list with OpenAI: {e}")
            st.session_state.prompt_generation_done = False

with col2:
    if st.button("Clear Prompts", use_container_width=True):
        reset_prompt_inputs(int(page_count))
        st.session_state.page_prompts = []
        st.session_state.prompt_generation_done = False
        st.success("Prompt list cleared.")

# =========================================================
# Step 2: Review / Edit Prompts
# =========================================================

if st.session_state.page_prompts:
    st.subheader("Review / Edit Page Prompts")
    st.caption("You can edit any prompt before image generation starts.")

    edited_prompts = []

    for i, prompt in enumerate(st.session_state.page_prompts):
        edited_value = st.text_input(
            f"Page {i + 1}",
            value=prompt,
            key=f"prompt_{i}"
        )
        edited_prompts.append(edited_value.strip())

    st.session_state.page_prompts = edited_prompts

    bad_prompts = [
        p for p in edited_prompts
        if not p or "background" in p.lower() or "multiple" in p.lower()
    ]

    if bad_prompts:
        st.warning("Some prompts may need review. Empty prompts or prompts mentioning background/multiple may give poor coloring-book results.")

# =========================================================
# Step 3: Generate Images and Build PDF
# =========================================================

if st.session_state.page_prompts:
    if st.button("Generate Images & PDF", type="primary", use_container_width=True):
        final_prompts = [p.strip() for p in st.session_state.page_prompts if p.strip()]

        if len(final_prompts) != int(page_count):
            st.error(f"You need exactly {int(page_count)} non-empty prompts before generating images.")
            st.stop()

        pdf = FPDF(unit="in", format=(8.5, 11))
        progress_bar = st.progress(0)
        status_text = st.empty()

        quality_wrapper = "300 dpi style"

        for i, page_subject in enumerate(final_prompts):
            current_page = i + 1
            status_text.text(f"Generating Page {current_page} of {page_count}: {page_subject}")

            try:
                final_prompt = f"{page_subject}, {quality_wrapper}, {STYLE_SUFFIX}"

                response = together_client.images.generate(
                    prompt=final_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    model=together_model,
                    width=int(image_width),
                    height=int(image_height),
                    steps=int(image_steps),
                    n=1
                )

                img_url = response.data[0].url
                img_data = requests.get(img_url, timeout=60).content

                pdf.add_page()
                pdf.image(BytesIO(img_data), x=0.5, y=0.5, w=7.5)

                del img_data
                gc.collect()

            except Exception as e:
                st.error(f"Error on page {current_page} ({page_subject}): {e}")
                break

            progress_bar.progress(current_page / len(final_prompts))

        status_text.text("Book assembly complete.")

        try:
            pdf_bytes = bytes(pdf.output())

            st.download_button(
                label="Download KDP-Ready PDF",
                data=pdf_bytes,
                file_name=f"{book_title.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Could not build PDF: {e}")
