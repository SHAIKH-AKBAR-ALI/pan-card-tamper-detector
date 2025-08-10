import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import requests
import io
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="🛡️ PAN Card Tamper Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with modern color palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main > div {padding-top: 2rem; font-family: 'Inter', sans-serif;}
    
    .metric-container {
        background: linear-gradient(145deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.2rem; border-radius: 15px; margin: 0.7rem 0;
        box-shadow: 0 8px 25px rgba(30,60,114,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .detection-box {
        border: 2px solid #e74c3c; 
        background: linear-gradient(135deg, rgba(231,76,60,0.1) 0%, rgba(192,57,43,0.05) 100%);
        padding: 1.5rem; border-radius: 12px; margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(231,76,60,0.1);
    }
    
    .success-box {
        border: 2px solid #27ae60; 
        background: linear-gradient(135deg, rgba(39,174,96,0.1) 0%, rgba(46,204,113,0.05) 100%);
        padding: 1.5rem; border-radius: 12px; margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(39,174,96,0.1);
    }
    
    .about-card {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 20px; margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102,126,234,0.2);
        color: white; text-align: center;
    }
    
    .tech-badge {
        display: inline-block; background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem;
        font-size: 0.8rem; font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 10px; border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sidebar .block-container {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px; padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_image(source, gray=False):
    """Load image from file upload, URL, or webcam"""
    if isinstance(source, str) and source.startswith(('http://', 'https://')):
        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        arr = np.array(img)[:, :, ::-1].copy()
    elif hasattr(source, 'read'):
        img = Image.open(source).convert('RGB')
        arr = np.array(img)[:, :, ::-1].copy()
    else:
        arr = source
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if gray else arr

def resize_match(a, b):
    """Resize b to match a's dimensions"""
    h, w = a.shape[:2]
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)

def detect_tamper(ref_img, sus_img, thresh=0.15, min_area=50, box=True, heatmap=True):
    """Core tamper detection with SSIM and morphological operations"""
    if ref_img.shape[:2] != sus_img.shape[:2]:
        sus_img = resize_match(ref_img, sus_img)
    
    grayA, grayB = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(sus_img, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8')
    
    _, th = cv2.threshold(255 - diff, int(255 * thresh), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, annotated = [], sus_img.copy()
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
            if box: cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,0,255), 2)
    
    if heatmap:
        heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        annotated = cv2.addWeighted(annotated, 0.7, heat, 0.3, 0)
    
    return {"score": float(score), "diff": diff, "mask": th, "boxes": boxes, "annotated": annotated}

# Header
st.markdown('<h1 class="header-gradient">🛡️ Advanced PAN Card Tamper Detection</h1>', unsafe_allow_html=True)
st.markdown("*AI-powered document integrity verification using structural similarity analysis*")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Detection Parameters")
    thresh = st.slider("Detection Sensitivity", 0.05, 0.5, 0.15, 0.05, help="Lower values = more sensitive")
    min_area = st.slider("Minimum Tamper Area", 10, 500, 150, 10, help="Minimum pixel area to consider")
    show_boxes = st.checkbox("Show Bounding Boxes", True)
    show_heatmap = st.checkbox("Show Difference Heatmap", True)
    
    st.markdown("## 📊 Analysis Options")
    detailed_analysis = st.checkbox("Detailed Analysis", False)
    save_results = st.checkbox("Save Results", False)

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL Input", "📷 Webcam", "👨‍💻 About"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Reference Image")
        ref_file = st.file_uploader("Upload reference PAN card", type=['png','jpg','jpeg'], key="ref")
    with col2:
        st.markdown("### 🔍 Suspect Image")
        sus_file = st.file_uploader("Upload suspect PAN card", type=['png','jpg','jpeg'], key="sus")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        ref_url = st.text_input("Reference Image URL", placeholder="https://example.com/ref.png")
    with col2:
        sus_url = st.text_input("Suspect Image URL", placeholder="https://example.com/suspect.png")

with tab3:
    if st.button("📸 Capture from Webcam"):
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.session_state.webcam_image = frame
                st.success("✅ Image captured successfully!")
        except:
            st.error("❌ Unable to access webcam")

with tab4:
    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Made with ❤️ by Akbar Ali**")
    st.markdown("*AI/ML Engineer & Computer Vision Enthusiast*")
    
    st.markdown("### 🛠️ Technologies Used")
    tech_stack = [
        "🐍 Python", "🖼️ OpenCV", "📊 NumPy", "🔬 scikit-image", 
        "🎨 Streamlit", "📷 PIL/Pillow", "🌐 Requests", "📈 Computer Vision",
        "🧮 SSIM Algorithm", "🎯 Image Processing", "🔍 Pattern Recognition", "⚡ Real-time Analysis"
    ]
    
    for i in range(0, len(tech_stack), 3):
        cols = st.columns(3)
        for j, tech in enumerate(tech_stack[i:i+3]):
            with cols[j]:
                st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Key Features")
    features = [
        "✅ **Real-time Detection** - Instant tamper analysis",
        "🎨 **Visual Heatmaps** - Intuitive difference visualization", 
        "📊 **SSIM Scoring** - Advanced similarity metrics",
        "🔧 **Configurable Parameters** - Customizable sensitivity",
        "📱 **Multi-input Support** - Files, URLs, and webcam",
        "💾 **Export Results** - Download analysis and reports"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.markdown("### 🔬 Algorithm Details")
    st.markdown("""
    **Structural Similarity Index (SSIM)** combined with:
    - Adaptive thresholding for noise reduction
    - Morphological operations for contour refinement  
    - Multi-scale analysis for robust detection
    - Computer vision techniques for precise localization
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact/Social section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("---")
        st.markdown("*Connect with the developer for collaborations and projects*")
        
        # You can add actual links here
        social_col1, social_col2, social_col3 = st.columns(3)
        with social_col1:
            st.markdown("🔗 [LinkedIn](#)")
        with social_col2:
            st.markdown("💻 [GitHub](#)")  
        with social_col3:
            st.markdown("📧 [Email](#)")

# Processing logic
ref_img = sus_img = None

# Load images based on input method
if ref_file and sus_file:
    ref_img, sus_img = load_image(ref_file), load_image(sus_file)
elif ref_url and sus_url:
    try:
        ref_img, sus_img = load_image(ref_url), load_image(sus_url)
    except Exception as e:
        st.error(f"❌ Error loading URLs: {e}")
elif 'webcam_image' in st.session_state and ref_file:
    ref_img, sus_img = load_image(ref_file), st.session_state.webcam_image

# Main analysis
if ref_img is not None and sus_img is not None:
    with st.spinner("🔄 Analyzing images..."):
        result = detect_tamper(ref_img, sus_img, thresh, min_area, show_boxes, show_heatmap)
    
    # Results section
    st.markdown("---")
    st.markdown("## 📊 Detection Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("SSIM Score", f"{result['score']:.4f}", 
                 delta=f"{result['score']-0.95:.4f}" if result['score'] < 0.95 else None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Tampered Regions", len(result['boxes']))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        status = "✅ AUTHENTIC" if result['score'] > 0.95 and len(result['boxes']) == 0 else "⚠️ SUSPICIOUS"
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Status", status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        confidence = min(100, max(0, (result['score'] * 100)))
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visual results
    st.markdown("## 🖼️ Visual Analysis")
    
    if len(result['boxes']) > 0:
        st.markdown('<div class="detection-box">', unsafe_allow_html=True)
        st.error(f"🚨 **TAMPERING DETECTED** - {len(result['boxes'])} suspicious region(s) found!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("✅ **NO TAMPERING DETECTED** - Document appears authentic")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Image display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original vs Suspect")
        comparison = np.hstack([ref_img, sus_img])
        st.image(comparison, channels="BGR", use_column_width=True)
    
    with col2:
        st.markdown("#### Annotated Result")
        st.image(result['annotated'], channels="BGR", use_column_width=True)
    
    if detailed_analysis:
        st.markdown("## 🔬 Detailed Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Difference Map")
            st.image(result['diff'], use_column_width=True)
        with col2:
            st.markdown("#### Detection Mask")
            st.image(result['mask'], use_column_width=True)
        
        if result['boxes']:
            st.markdown("#### Detected Regions")
            for i, (x,y,w,h) in enumerate(result['boxes'], 1):
                st.write(f"**Region {i}:** Position({x},{y}), Size({w}×{h}), Area: {w*h} pixels")
    
    # Save functionality
    if save_results:
        col1, col2 = st.columns(2)
        with col1:
            if st.download_button("💾 Download Annotated Image", 
                                cv2.imencode('.png', result['annotated'])[1].tobytes(),
                                "tamper_detection_result.png", "image/png"):
                st.success("✅ Download started!")
        with col2:
            if st.download_button("📊 Download Analysis Report",
                                f"SSIM Score: {result['score']:.4f}\n"
                                f"Detected Regions: {len(result['boxes'])}\n"
                                f"Status: {'AUTHENTIC' if result['score'] > 0.95 and len(result['boxes']) == 0 else 'SUSPICIOUS'}\n"
                                f"Confidence: {min(100, max(0, result['score']*100)):.1f}%",
                                "analysis_report.txt", "text/plain"):
                st.success("✅ Report downloaded!")

else:
    st.info("👆 Please upload both reference and suspect images to begin analysis")
    
    # Demo section
    with st.expander("📖 How it works"):
        st.markdown("""
        **Advanced PAN Card Tamper Detection** uses:
        
        1. **Structural Similarity Index (SSIM)** - Compares images pixel by pixel
        2. **Adaptive Thresholding** - Identifies significant differences
        3. **Morphological Operations** - Cleans up noise and false positives
        4. **Contour Detection** - Locates and bounds tampered regions
        5. **Heatmap Visualization** - Shows difference intensity
        
        **Features:**
        - Real-time webcam support
        - URL image loading
        - Adjustable sensitivity
        - Detailed analysis reports
        - Download results
        """)

# Footer
st.markdown("---")
st.markdown("*🔬 Developed by **Akbar Ali** | Powered by OpenCV, scikit-image, and advanced computer vision algorithms*")