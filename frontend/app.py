import streamlit as st
import requests
from io import BytesIO
from config import settings



st.set_page_config(page_title="PDF Uploader", page_icon="📄")

st.title("📄 PDF Uploader and Processor")

st.markdown("""
Welcome to the PDF Uploader and Processor! 
Please upload a PDF file to extract its text content.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Select a PDF file to upload and process.")

if uploaded_file is not None:
    # Read the file into a BytesIO object
    file_bytes = BytesIO(uploaded_file.read())

    # Display file details in a formatted way
    st.subheader("File Details")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**File Size:** {len(file_bytes.getvalue())} bytes")

    # Display an image if the PDF contains one
    st.subheader("Preview")
    st.write("This section will show a preview of the first page if the PDF contains any images.")

    # Trigger the backend API
    if st.button("Upload and Process PDF", help="Click to upload and process the PDF file."):
        with st.spinner("Processing..."):
            
            # Prepare the API request
            # url = settings.LOCAL_BACKEND_SERVICE_URL
            url = settings.BACKEND_SERVICE_URL

            
            files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
            
            # Send the request
            response = requests.post(url, files=files)
            
            # Check the response
            if response.status_code == 200:
                st.success("PDF processed successfully!")
                st.subheader("Extracted Text")
                st.text(response.json().get("text", "No text extracted."))
            else:
                st.error(f"Failed to process PDF. Status code: {response.status_code}")
                st.write(response.text)
