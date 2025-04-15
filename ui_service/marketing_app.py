import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import json
from config import settings

st.set_page_config(
    page_title="Marketing Content Generator",
    page_icon="📢",
    layout="wide"
)

# Constants for marketing options
PLATFORMS = [
    "Facebook", "Instagram", "Twitter", "LinkedIn", "Email", "Blog", 
    "Website", "Pinterest", "TikTok", "YouTube"
]

TONES = [
    "professional", "friendly", "enthusiastic", "authoritative", 
    "humorous", "inspirational", "educational", "promotional", "conversational"
]

FORMATS = [
    "social media post", "ad copy", "product description", "blog post", 
    "email newsletter", "promotional email", "blog article", "press release",
    "landing page copy", "video script", "infographic text"
]

CONTENT_LENGTHS = [
    "very short (1-2 sentences)", 
    "short (1 paragraph)", 
    "medium (2-3 paragraphs)", 
    "long (article length)"
]

def display_marketing_result(response_data):
    """Display the retrieved content and generated marketing content"""
    if response_data:
        # Create tabs for different types of content
        tabs = st.tabs(["Generated Marketing Content", "Knowledge Base", "Image Resources", "Prompt"])
        
        # Tab 1: Generated Marketing Content
        with tabs[0]:
            if "answer" in response_data and response_data["answer"]:
                st.markdown("## Generated Marketing Content")
                st.markdown(response_data["answer"])
                
                # Offer download options
                marketing_content = response_data["answer"]
                st.download_button(
                    label="Download as Text",
                    data=marketing_content,
                    file_name="marketing_content.txt",
                    mime="text/plain"
                )
                
                # Add copy button - implemented via JavaScript
                st.markdown("""
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px;">
                    <p style="font-weight:bold">Copy the marketing content to clipboard:</p>
                    <button 
                        onclick="
                            navigator.clipboard.writeText(document.getElementById('content-to-copy').innerText);
                            this.innerText='Copied!';
                            setTimeout(() => this.innerText='Copy to Clipboard', 2000);
                        "
                        style="background-color:#4CAF50;color:white;padding:8px 16px;border:none;border-radius:4px;cursor:pointer;"
                    >
                        Copy to Clipboard
                    </button>
                </div>
                <div id="content-to-copy" style="display:none;">
                """ + marketing_content + """
                </div>
                """, unsafe_allow_html=True)
                
                # Sharing hints based on platform
                platform = response_data.get("marketing_content", {}).get("platform", "")
                if platform:
                    st.markdown(f"### Sharing Tips for {platform}")
                    
                    if platform.lower() == "facebook":
                        st.info("Consider adding relevant hashtags and a compelling image to increase engagement.")
                    elif platform.lower() == "instagram":
                        st.info("Use 5-10 targeted hashtags and ensure your image is high quality and visually appealing.")
                    elif platform.lower() == "twitter":
                        st.info("Keep it concise, use 1-2 relevant hashtags, and consider adding a visual element.")
                    elif platform.lower() == "linkedin":
                        st.info("Add 3-5 industry-specific hashtags and consider including statistics or data points.")
                    elif platform.lower() == "email":
                        st.info("Use a compelling subject line and ensure your content is mobile-friendly.")
            else:
                st.info("No marketing content was generated. Try adjusting your query or options.")
        
        # Tab 2: Knowledge Base Content
        with tabs[1]:
            text_results = response_data.get("text_results", [])
            if text_results:
                st.markdown("## Knowledge Base Information Used")
                for i, item in enumerate(text_results):
                    with st.expander(f"Source {i+1} - Relevance: {item.get('score', 0):.2f}"):
                        st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                        st.markdown(f"**Content:**\n{item.get('content', 'No content')}")
            else:
                st.info("No text content was retrieved from the knowledge base.")
        
        # Tab 3: Image Resources
        with tabs[2]:
            image_results = response_data.get("image_results", [])
            if image_results:
                st.markdown("## Visual Resources Used")
                
                # Create a grid layout for images
                cols = st.columns(min(3, len(image_results)))
                
                for i, item in enumerate(image_results):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        st.markdown(f"**Relevance:** {item.get('score', 0):.2f}")
                        
                        # Display image if available
                        if "base64_data" in item:
                            try:
                                if "placeholder_" in item["base64_data"]:
                                    st.info(f"Image placeholder: {item.get('image_id', 'Unknown')}")
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
                        if item.get('source'):
                            st.markdown(f"**Source:** {item.get('source')}")
            else:
                st.info("No image resources were retrieved.")
        
        # Tab 4: Prompt Information
        with tabs[3]:
            st.markdown("## Prompt Details")
            st.markdown("### Context Used")
            st.text(response_data.get("context", "No context provided"))
            
            st.markdown("### Generated Prompt")
            st.text(response_data.get("prompt", "No prompt generated"))

def main():
    st.title("📢 Marketing Content Generator")
    st.markdown("""
    Use AI to generate high-quality marketing content for your small business.
    This tool leverages our knowledge base to create accurate and engaging content for different platforms.
    """)
    
    # Create a two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "What marketing content do you need?", 
            height=100,
            placeholder="Example: Write a post about our new eco-friendly product line highlighting the sustainability benefits",
            help="Describe the marketing content you need in detail. Include product info, target audience, and key points."
        )
        
        with st.expander("Advanced Content Options", expanded=True):
            # First row of options
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                platform = st.selectbox("Platform", PLATFORMS, index=0)
                tone = st.selectbox("Tone", TONES, index=0)
            
            with opt_col2:
                content_format = st.selectbox("Format", FORMATS, index=0)
                content_length = st.selectbox("Length", CONTENT_LENGTHS, index=1)
            
            target_audience = st.text_input(
                "Target Audience", 
                value="small business owners",
                help="Describe your target audience (e.g., young professionals, parents, tech enthusiasts)"
            )
            
    with col2:
        st.markdown("### Knowledge Base Options")
        k_text = st.slider("Number of text sources to use", min_value=1, max_value=10, value=5)
        k_images = st.slider("Number of image sources to use", min_value=0, max_value=5, value=2)
        
        image_query = st.text_input(
            "Specific image search (optional)",
            placeholder="e.g., product images, team photos",
            help="Optional: provide a specific description for image search. If empty, the main query will be used."
        )
        
        # Add model selection
        model_option = st.selectbox(
            "AI model:",
            ("LLaVA (text + images)", "Llama 2 (text only)"),
            index=0
        )
        
        use_llava = model_option == "LLaVA (text + images)"
    
    # Add visual divider
    st.markdown("---")
    
    # Process generation button
    if st.button("Generate Marketing Content", type="primary"):
        if query:
            with st.spinner("Generating marketing content..."):
                # Prepare the request payload with marketing options
                marketing_options = {
                    "enabled": True,
                    "platform": platform,
                    "target_audience": target_audience,
                    "tone": tone,
                    "format": content_format,
                    "content_length": content_length
                }
                
                payload = {
                    "query": query,
                    "k_text": int(k_text),
                    "k_images": int(k_images),
                    "include_image_data": True,
                    "generate_answer": True,
                    "use_llava": use_llava,
                    "marketing_options": marketing_options
                }
                
                # Add image query if provided
                if image_query:
                    payload["image_query"] = image_query
                
                # Use the marketing endpoint
                try:
                    response = requests.post(
                        settings.INFERENCE_SERVICE_URL.replace("/query", "/marketing"),
                        json=payload,
                        timeout=60  # Increase timeout for marketing content generation
                    )
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        result = response.json()
                        display_marketing_result(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a description of the marketing content you need.")

    # Example templates section
    with st.expander("Marketing Templates and Examples"):
        template_tabs = st.tabs(["Social Media", "Email", "Blog Posts", "Ad Copy"])
        
        with template_tabs[0]:
            st.markdown("""
            ### Social Media Templates
            
            #### Product Launch
            *Generate a social media post announcing our new [product name] that highlights its key features and benefits for [target audience]*
            
            #### Testimonial Highlight
            *Create a social media post featuring a customer testimonial about our [product/service] emphasizing the positive results they achieved*
            
            #### Event Promotion
            *Write a social media post promoting our upcoming [event name] on [date], highlighting why people should attend and how to register*
            """)
            
        with template_tabs[1]:
            st.markdown("""
            ### Email Templates
            
            #### Newsletter
            *Create an email newsletter about our recent company achievements, new [product/service] offerings, and upcoming events for our [customer type] audience*
            
            #### Promotional Offer
            *Write an email promoting our limited-time [discount/offer] on [product/service], emphasizing the value and creating urgency*
            
            #### Follow-up Email
            *Create a follow-up email to send to prospects who showed interest in our [product/service] during [event/webinar], addressing common questions*
            """)
            
        with template_tabs[2]:
            st.markdown("""
            ### Blog Post Templates
            
            #### How-To Guide
            *Write a blog post titled "How to [solve problem] with [your product/service]" that walks readers through the process step-by-step*
            
            #### Industry Trends
            *Create a blog post about the top trends in [your industry] for [year/season] and how our [products/services] align with these trends*
            
            #### Case Study
            *Write a blog post highlighting how our [product/service] helped [client/customer type] achieve [specific results], including the challenges they faced*
            """)
            
        with template_tabs[3]:
            st.markdown("""
            ### Ad Copy Templates
            
            #### Google Ads
            *Write Google ad copy for our [product/service] with a compelling headline and description that emphasizes [key benefit] for [target audience]*
            
            #### Facebook Ad
            *Create Facebook ad copy promoting our [product/service/offer] that grabs attention and includes a clear call-to-action for [target audience]*
            
            #### Product Description
            *Write a compelling product description for our [product name] highlighting its features, benefits, and why it's perfect for [target audience]*
            """)

if __name__ == "__main__":
    main()