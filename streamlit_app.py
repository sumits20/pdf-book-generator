import streamlit as st
from openai import OpenAI
from fpdf import FPDF
import requests
from io import BytesIO
from PIL import Image

st.title("🎨 AI KDP Book Creator")
st.subheader("Generate ready-to-upload PDFs for Amazon")

# Sidebar for API Key & Settings
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    book_title = st.text_input("Book Title", "My Coloring Book")
    page_count = st.number_input("Number of Pages", min_value=1, max_value=50, value=5)
    
if not api_key:
    st.warning("Please enter your OpenAI API Key to start.")
    st.stop()

client = OpenAI(api_key=api_key)

# The Logic: Generate PDF
if st.button(f"Generate {page_count} Page Book"):
    pdf = FPDF(unit="in", format=(8.5, 11))
    
    progress_bar = st.progress(0)
    
    for i in range(page_count):
        st.write(f"Generating Page {i+1}...")
        
        # 1. API Call (Using DALL-E 3)
        # Note: We use a placeholder prompt for now
        response = client.images.generate(
            model="dall-e-3",
            prompt="Simple kids coloring book page, clean black lines, white background, cute animal",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        img_url = response.data[0].url
        img_data = requests.get(img_url).content
        
        # 2. Add to PDF
        pdf.add_page()
        # We use BytesIO to avoid saving files to the cloud disk
        pdf.image(BytesIO(img_data), x=0.5, y=0.5, w=7.5)
        
        progress_bar.progress((i + 1) / page_count)

    # 3. Final Export
    pdf_output = pdf.output() # Returns bytes in fpdf2
    st.success("✅ Book Generated!")
    st.download_button(
        label="Download KDP PDF",
        data=pdf_output,
        file_name=f"{book_title}.pdf",
        mime="application/pdf"
    )
