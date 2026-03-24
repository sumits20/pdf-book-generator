import streamlit as st
from together import Together
from openai import OpenAI
from xai_sdk import Client as XAIClient
from fpdf import FPDF
import requests
from io import BytesIO
import gc
import json
import re
import base64

# =========================================================
# Session State
# =========================================================

if "page_prompts" not in st.session_state:
    st.session_state.page_prompts = []

# =========================================================
# Prompt / Style Settings
# =========================================================

STYLE_SUFFIX = (
    "cute toddler coloring book page, ages 3 to 6, "
    "simple cartoon style, one subject only, centered on page, "
    "full body visible, big friendly eyes, "
    "thick bold clean black outlines, uniform line thickness, "
    "minimal interior detail, minimal fur or feather texture, "
    "large easy coloring spaces, plain white background, "
    "black and white line art only",
    "Add border, no bleeding outside border"
)

NEGATIVE_PROMPT = (
    "color, coloured, grayscale, grey fill, gray fill, shading, gradients, shadows, lighting, "
    "dark background, black background, realistic, semi-realistic, "
    "detailed fur, detailed feathers, texture, sketch, messy lines, thin lines, blur, noise, "
    "cross-hatching, hatching, text, watermark, multiple characters, crowd, extra animals, toys"
)

# =========================================================
# Streamlit Page
# =========================================================

st.set_page_config(page_title="Multi-Provider KDP Generator", page_icon="📚")
st.title("AI Powered KDP Book Maker")
st.markdown("Generate prompts with OpenAI, review them, then create images using OpenAI, Together AI, or Grok.")

# =========================================================
# API Clients
# =========================================================

try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    openai_client = None

try:
    together_client = Together(api_key=st.secrets["TOGETHER_API_KEY"])
except Exception:
    together_client = None

try:
    xai_client = XAIClient(api_key=st.secrets["XAI_API_KEY"])
except Exception:
    xai_client = None

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("Book Settings")

    book_title = st.text_input("Book Filename", "my_coloring_book")
    page_count = st.number_input("Number of Pages", min_value=1, max_value=100, value=10)

    st.divider()
    st.subheader("Theme Input")
    master_prompt = st.text_area("Book Theme", "Baby Animals")

    st.divider()
    st.subheader("Image Provider")

    image_provider = st.selectbox(
        "Choose image provider",
        ["Together AI", "OpenAI", "Grok"]
    )

    if image_provider == "Together AI":
        together_model = st.text_input(
            "Together Model",
            "stabilityai/stable-diffusion-xl-base-1.0"
        )
        image_steps = st.slider("Image Steps", min_value=20, max_value=60, value=30)
        image_width = st.number_input("Image Width", min_value=512, max_value=1024, value=800, step=64)
        image_height = st.number_input("Image Height", min_value=512, max_value=1408, value=1024, step=64)

    elif image_provider == "OpenAI":
        openai_image_model = st.selectbox(
            "OpenAI Image Model",
            ["gpt-image-1-mini", "gpt-image-1"]
        )
        openai_image_size = st.selectbox(
            "OpenAI Image Size",
            ["1024x1024", "1024x1536", "1536x1024"],
            index=1
        )
        openai_image_quality = st.selectbox(
            "OpenAI Image Quality",
            ["low", "medium", "high"],
            index=1
        )

    elif image_provider == "Grok":
        grok_image_model = st.text_input(
            "Grok Image Model",
            "grok-imagine-image"
        )

# =========================================================
# Helpers
# =========================================================

def extract_json_array(text: str):
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
    if not openai_client:
        raise ValueError("OPENAI_API_KEY is missing. OpenAI is required for prompt generation.")

    instructions = (
        "You create short prompt items for a toddler-friendly children's coloring book. "
        "Return ONLY a valid JSON array of strings. "
        "Each item must describe exactly one cute cartoon subject suitable for a simple coloring page. "
        "Avoid realistic wording, backgrounds, scenery, and multiple subjects."
    )

    user_input = f"""
Theme: {theme}
Number of pages: {count}

Create exactly {count} short coloring-book prompts.

Rules:
- Make every subject cute, cartoon-like, and toddler-friendly.
- Prefer big eyes, round face, smiling expression, simple pose.
- Keep prompts short.
- Avoid realistic wording.
- Good examples:
  "cute baby elephant cartoon, big eyes, sitting"
  "cute baby bunny cartoon, smiling, front view"
  "cute toy car with eyes, smiling, front view"
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
        raise ValueError(f"Expected {count} prompts, but got {len(cleaned)}.\nParsed prompts: {cleaned}")

    return cleaned


def generate_image_bytes(provider: str, prompt: str) -> bytes:
    """
    Return raw image bytes from the selected provider.
    """

    if provider == "Together AI":
        if not together_client:
            raise ValueError("TOGETHER_API_KEY is missing.")

        response = together_client.images.generate(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            model=together_model,
            width=int(image_width),
            height=int(image_height),
            steps=int(image_steps),
            n=1
        )
        img_url = response.data[0].url
        return requests.get(img_url, timeout=60).content

    elif provider == "OpenAI":
        if not openai_client:
            raise ValueError("OPENAI_API_KEY is missing.")

        response = openai_client.images.generate(
            model=openai_image_model,
            prompt=prompt,
            size=openai_image_size,
            quality=openai_image_quality,
            output_format="png"
        )

        b64_img = response.data[0].b64_json
        return base64.b64decode(b64_img)

    elif provider == "Grok":
        if not xai_client:
            raise ValueError("XAI_API_KEY is missing.")

        response = xai_client.image.sample(
            prompt=prompt,
            model=grok_image_model,
        )

        img_url = response.url
        return requests.get(img_url, timeout=60).content

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# =========================================================
# Step 1: Generate Prompt List
# =========================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Prompt List", use_container_width=True):
        try:
            with st.spinner("Creating page prompt list with OpenAI..."):
                st.session_state.page_prompts = build_page_prompt_list(master_prompt, int(page_count))
            st.success("Prompt list created successfully.")
        except Exception as e:
            st.error(f"Failed to generate prompt list: {e}")

with col2:
    if st.button("Clear Prompts", use_container_width=True):
        st.session_state.page_prompts = []
        for i in range(200):
            key = f"prompt_{i}"
            if key in st.session_state:
                del st.session_state[key]
        st.success("Prompt list cleared.")

# =========================================================
# Step 2: Review / Edit Prompts
# =========================================================

if st.session_state.page_prompts:
    st.subheader("Review / Edit Page Prompts")

    edited_prompts = []
    for i, prompt in enumerate(st.session_state.page_prompts):
        edited_value = st.text_input(
            f"Page {i + 1}",
            value=prompt,
            key=f"prompt_{i}"
        )
        edited_prompts.append(edited_value.strip())

    st.session_state.page_prompts = edited_prompts

# =========================================================
# Step 3: Generate Images and Build PDF
# =========================================================

if st.session_state.page_prompts:
    if st.button("Generate Images & PDF", type="primary", use_container_width=True):
        final_prompts = [p.strip() for p in st.session_state.page_prompts if p.strip()]

        if len(final_prompts) != int(page_count):
            st.error(f"You need exactly {int(page_count)} non-empty prompts.")
            st.stop()

        pdf = FPDF(unit="in", format=(8.5, 11))
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page_subject in enumerate(final_prompts):
            current_page = i + 1
            status_text.text(f"Generating Page {current_page} of {page_count}: {page_subject}")

            try:
                final_prompt = f"{page_subject}, {STYLE_SUFFIX}"
                img_data = generate_image_bytes(image_provider, final_prompt)

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
