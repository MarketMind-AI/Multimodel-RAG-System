import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import json
from config import settings

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multimodal RAG Query System")
st.markdown("""
This system can search and retrieve both text and images from your document database. 
Ask any question about your documents and see relevant text passages and images!
""")

# Create a two-column layout for the query inputs
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_area("Your query:", height=100, 
                         help="Enter your main query here. This will be used for both text and image search.")

with col2:
    image_query = st.text_input("Image query (optional):", 
                              help="Optional: provide a specific description for image search. If empty, the main query will be used.")
    k_text = st.number_input("Number of text results:", min_value=1, max_value=10, value=5)
    k_images = st.number_input("Number of image results:", min_value=0, max_value=10, value=3)

# Add a toggle for including actual image data
include_images = st.checkbox("Include image data in results", value=True, 
                            help="When checked, the system will retrieve and display the actual images. Uncheck for faster performance.")

# Add model selection
model_option = st.selectbox(
    "Select LLM model:",
    ("LLaVA (multimodal)", "Llama 2 (text-only)")
)

# Process query button
if st.button("Search Documents"):
    if query:
        with st.spinner("Searching for relevant content..."):
            # Prepare the request payload
            payload = {
                "query": query,
                "k_text": int(k_text),
                "k_images": int(k_images),
                "include_image_data": include_images
            }
            
            # Add image query if provided
            if image_query:
                payload["image_query"] = image_query
            
            # Make the request to the inference service
            try:
                response = requests.post(
                    settings.INFERENCE_SERVICE_URL,
                    json=payload,
                    timeout=30
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the context
                    st.subheader("Retrieved Information")
                    
                    # Create tabs for different types of content
                    tabs = st.tabs(["Text Results", "Image Results", "Generated Response"])
                    
                    # Display text results
                    with tabs[0]:
                        text_results = result.get("text_results", [])
                        if text_results:
                            for i, item in enumerate(text_results):
                                with st.expander(f"Text Result {i+1} - Score: {item['score']:.4f}"):
                                    st.markdown(f"**Source:** {item['source']}")
                                    if 'page_range' in item:
                                        st.markdown(f"**Pages:** {item['page_range']}")
                                    st.markdown("**Content:**")
                                    st.markdown(item['content'])
                        else:
                            st.info("No text results found.")
                    
                    # Display image results
                    with tabs[1]:
                        image_results = result.get("image_results", [])
                        if image_results:
                            # Create a grid layout for images
                            cols = st.columns(min(3, len(image_results)))
                            
                            for i, item in enumerate(image_results):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.markdown(f"**Score:** {item['score']:.4f}")
                                    
                                    # If image data is included
                                    if include_images and "base64_data" in item:
                                        try:
                                            # If it's a placeholder (development)
                                            if "placeholder_" in item["base64_data"]:
                                                st.info(f"Image: {item['image_id']}")
                                            else:
                                                # Extract the actual base64 data
                                                img_data = item["base64_data"].split(",")[1]
                                                binary_data = base64.b64decode(img_data)
                                                img = Image.open(BytesIO(binary_data))
                                                st.image(img, caption=item.get('caption', 'No caption'))
                                        except Exception as e:
                                            st.error(f"Error displaying image: {str(e)}")
                                    
                                    # Display metadata
                                    st.markdown(f"**Caption:** {item.get('caption', 'No caption')}")
                                    st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                                    if item.get('page_num'):
                                        st.markdown(f"**Page:** {item['page_num']}")
                        else:
                            st.info("No image results found.")
                    
                    # Generate and display the AI response
                    with tabs[2]:
                        prompt = result.get("prompt", "")
                        context = result.get("context", "")
                        
                        with st.expander("View Context and Prompt"):
                            st.subheader("Context")
                            st.text(context)
                            
                            st.subheader("Generated Prompt")
                            st.text(prompt)
                        
                        st.subheader("AI Response:")
                        
                        # Call the appropriate LLM service based on user selection
                        with st.spinner("Generating response..."):
                            try:
                                if model_option == "LLaVA (multimodal)" and include_images and result.get("image_results"):
                                    # Call multimodal endpoint with both text and images
                                    llm_payload = {
                                        "prompt": query,
                                        "context": context,
                                        "text_results": result.get("text_results", []),
                                        "image_results": result.get("image_results", [])
                                    }
                                    
                                    llm_response = requests.post(
                                        settings.MULTIMODAL_LLM_SERVICE_URL,
                                        json=llm_payload,
                                        timeout=120  # Multimodal generation may take longer
                                    )
                                else:
                                    # Call text-only endpoint with just the context
                                    llm_payload = {
                                        "prompt": query,
                                        "context": context
                                    }
                                    
                                    llm_response = requests.post(
                                        settings.TEXT_LLM_SERVICE_URL,
                                        json=llm_payload,
                                        timeout=30
                                    )
                                
                                if llm_response.status_code == 200:
                                    llm_result = llm_response.json()
                                    answer = llm_result.get("answer", "No response generated.")
                                    st.write(answer)
                                else:
                                    st.error(f"Error generating response: {llm_response.status_code} - {llm_response.text}")
                                    st.info("Falling back to showing the retrieved context only.")
                            except Exception as e:
                                st.error(f"Error connecting to LLM service: {str(e)}")
                                st.info("For testing, you can review the retrieved text and images above.")
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")

# Add a simple example section at the bottom
with st.expander("Example Queries"):
    st.markdown("""
    Try some of these example queries:
    - "Explain the architecture diagram in the documents"
    - "How does the data flow from MongoDB to Qdrant?"
    - "Show me any diagrams related to the RAG system"
    - "What is the purpose of the bytewax component?"
    - "Can you analyze and describe the architecture of this system?"
    """)