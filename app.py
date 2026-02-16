import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ------------------------------
# Load model (cached) - FIXED
# ------------------------------
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_summarizer()

st.title("📄 PDF Notes Summarizer")

# ------------------------------
# Extract text from PDF
# ------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return " ".join(text.split())

# ------------------------------
# Clean text
# ------------------------------
def clean_text(text):
    import re
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[•●○■□▪▫]", "", text)
    return text.strip()

# ------------------------------
# Split text into chunks
# ------------------------------
def split_text(text, max_words=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

# ------------------------------
# Summarize text 
# ------------------------------
def summarize_text(text):
    text = clean_text(text)
    summaries = []
    chunks = split_text(text, max_words=400)
    
    for chunk in chunks:
        # Tokenize
        inputs = tokenizer(
            chunk,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=600,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Combine all summaries
    combined = " ".join(summaries)
    
    # If combined is too long, summarize again
    if len(combined.split()) > 500:
        inputs = tokenizer(
            combined,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=300,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return combined

# ------------------------------
# Streamlit UI
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    st.info(f"📖 Extracted {len(text.split())} words from PDF")
    
    if st.button("Generate Summary"):
        with st.spinner("AI is summarizing... (This may take a minute)"):
            summary = summarize_text(text)
        
        st.subheader("📝 Concise Summary")
        st.write(summary)
        
        # Stats
        original_words = len(text.split())
        summary_words = len(summary.split())
        reduction = int((1 - summary_words/original_words) * 100)
        st.success(f"✅ Reduced from {original_words} words to {summary_words} words ({reduction}% reduction)")
