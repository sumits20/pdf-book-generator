import streamlit as st
from together import Together
from fpdf import FPDF
import requests
from io import BytesIO
import gc

STYLE_SUFFIX = (
    "children's coloring book illustration, clean vector line art, "
    "centered composition, single subject, "
    "bold thick black outlines, uniform line thickness, crisp edges, "
    "pure white background, no background elements, "
    "large open spaces, minimal details, simple shapes, "
    "cute and child-friendly style, "
    "high contrast black and white, "
    "square border frame"
)

NEGATIVE_PROMPT = (
    "color, grayscale, shading, gradients, shadows, lighting, "
    "3D, realistic, textures, background, scenery, clutter, "
    "thin lines, sketch, messy lines, blur, noise, "
    "cross-hatching, hatching, complex details"
)


# 1. Setup Page Config
st.set_page_config(page_title="Together AI KDP Generator", page_icon="📚")

st.title("AI Powered KDP Book Maker")
st.markdown("Automate your coloring books for pennies per page.")

# 2. Initialize Together Client using Secret
try:
    client = Together(api_key=st.secrets["TOGETHER_API_KEY"])
except Exception as e:
    st.error("Missing TOGETHER_API_KEY in Secrets!")
    st.stop()

# 3. Sidebar UI
with st.sidebar:
    st.header("Book Settings")
    book_title = st.text_input("Book Filename", "my_together_book")
    page_count = st.number_input("Number of Pages", min_value=1, max_value=100, value=10)
    
    st.divider()
    st.subheader("Style Control")
    master_prompt = st.text_area(
        "Master Character/Theme", 
        "A cute baby elephant with big ears",
    )

# 4. The Generation Engine
if st.button(f"Generate {page_count}-Page PDF"):
    # Standard KDP 8.5x11 inches
    pdf = FPDF(unit="in", format=(8.5, 11))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # KDP-Specific Quality Prompt Wrapper
    quality_wrapper = "300 dpi style"

    for i in range(page_count):
        current_page = i + 1
        status_text.text(f"Generating Page {current_page} of {page_count}...")
        
        try:
            # Together AI API Call (SDXL 1.0 is the best 'cheap' model for lines)
            response = client.images.generate(
                # Combine user prompt with the bold style suffix
                prompt=f"{master_prompt}, {quality_wrapper}, {STYLE_SUFFIX}",
                negative_prompt=NEGATIVE_PROMPT, # This removes the grey shading
                model="stabilityai/stable-diffusion-xl-base-1.0",
                width=800,
                height=1024,
                steps=50, # Increased steps slightly for cleaner, smoother lines
                n=1
            )
            img_url = response.data[0].url
            img_data = requests.get(img_url).content
            
            # Add to PDF: Center image with 0.5" margins
            pdf.add_page()
            # FPDF2 allows passing BytesIO directly
            pdf.image(BytesIO(img_data), x=0.5, y=0.5, w=7.5)
            
            # MEMORY MANAGEMENT: Clear image data from RAM immediately
            del img_data
            gc.collect()
            
        except Exception as e:
            st.error(f"Error on page {current_page}: {str(e)}")
            break
            
        progress_bar.progress(current_page / page_count)

    # 5. Finalise and Download
    status_text.text("✅ Book Assembly Complete!")
    
    # FIX: Wrap the output in bytes() to make it compatible with Streamlit
    pdf_bytes = bytes(pdf.output()) 
    
    st.download_button(
        label="📥 Download KDP-Ready PDF",
        data=pdf_bytes,  # Now this is a standard bytes object
        file_name=f"{book_title.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
