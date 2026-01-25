"""
YOLO Vision AI - Object Detection Dashboard
Premium UI with Landing Page and Detection Interface
"""

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
from vehicle_counter import YOLOProcessor

# Page Config
st.set_page_config(
    page_title="YOLO Vision AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS - Organized in external file, loaded properly
def inject_global_css():
    """Load CSS from external file for better maintainability"""
    import os
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    
    # Load external CSS
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    except FileNotFoundError:
        css_content = ""  # Fallback if file missing
    
    # Inject Font Awesome and Google Fonts
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    # Inject CSS separately to avoid escaping issues
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ==================== LANDING PAGE ====================
def render_landing_page():
    """Render the premium landing page"""
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-icon">
            <i class="fa-solid fa-eye"></i>
        </div>
        <h1 class="hero-title">
            YOLO Vision <span style="color: var(--accent);">AI</span>
        </h1>
        <p class="hero-subtitle">
            Real-time object detection and counting powered by YOLO26
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection Cards
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode_col1, mode_col2 = st.columns(2, gap="medium")
        
        with mode_col1:
            st.markdown("""
            <div class="mode-card-container" data-target="vehicle-btn">
                <div class="mode-card">
                    <div class="mode-card-icon">🚗</div>
                    <div class="mode-card-title">Vehicle Counter</div>
                    <div class="mode-card-desc">Count cars, trucks, buses, and motorcycles in traffic footage</div>
                </div>
                <div class="mode-card-button-wrapper">
                    <button class="mode-card-button">SELECT VEHICLE MODE</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("", key="vehicle-btn", type="primary"):
                st.session_state.page = "dashboard"
                st.session_state.selected_mode = "Vehicle Counter"
                st.rerun()
        
        with mode_col2:
            st.markdown("""
            <div class="mode-card-container" data-target="person-btn">
                <div class="mode-card">
                    <div class="mode-card-icon">👥</div>
                    <div class="mode-card-title">Person Counter</div>
                    <div class="mode-card-desc">Track and count pedestrians in crowd footage</div>
                </div>
                <div class="mode-card-button-wrapper">
                    <button class="mode-card-button">SELECT PERSON MODE</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("", key="person-btn", type="primary"):
                st.session_state.page = "dashboard"
                st.session_state.selected_mode = "Person Counter"
                st.rerun()
    
    
    # Features Section
    st.markdown("""
    <div class="features">
        <div class="feature">
            <i class="fa-solid fa-bolt"></i>
            <span>Real-time Processing</span>
        </div>
        <div class="feature">
            <i class="fa-solid fa-chart-line"></i>
            <span>Cumulative Counting</span>
        </div>
        <div class="feature">
            <i class="fa-solid fa-microchip"></i>
            <span>YOLO26 Powered</span>
        </div>
        <div class="feature">
            <i class="fa-solid fa-fingerprint"></i>
            <span>ByteTrack IDs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== DASHBOARD ====================
def create_donut_chart(counts: dict) -> go.Figure:
    """Create class distribution donut chart"""
    if not counts or len(counts) <= 1:
        return None
    
    labels = []
    values = []
    for key, val in counts.items():
        if key != 'total' and val > 0:
            labels.append(key.title())
            values.append(val)
    
    if not labels:
        return None
        
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(line=dict(color='#fff', width=2)),
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=250,
        annotations=[dict(
            text=f'<b>{counts["total"]}</b>',
            x=0.5, y=0.5,
            font_size=24,
            showarrow=False
        )]
    )
    return fig

def render_dashboard():
    """Render the detection dashboard"""
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "landing"
        st.session_state.processing = False
        st.rerun()
    
    # Title
    st.markdown('<h1 style="text-align: center;"><i class="fa-solid fa-eye" style="color:var(--accent); margin-right:15px;"></i>YOLO Vision <span style="color:var(--accent);">AI</span></h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3><i class="fa-solid fa-sliders" style="margin-right:10px;"></i>Settings</h3>', unsafe_allow_html=True)
        
        # Mode Selection (pre-select from landing page if available)
        default_mode = st.session_state.get('selected_mode', 'Vehicle Counter')
        mode_options = ["Vehicle Counter", "Person Counter"]
        mode = st.selectbox(
            "Detection Mode",
            mode_options,
            index=mode_options.index(default_mode),
            help="Switch between vehicles or pedestrians"
        )
        
        config_name = "vehicle" if mode == "Vehicle Counter" else "person"
        
        st.markdown("---")
        
        # Video Source
        video_source_type = st.radio(
            "Video Source", 
            ["Sample Video", "Upload Video", "Webcam"],
            help="Select video source"
        )
        video_path = None
        use_camera = False
        camera_id = 0
        
        if video_source_type == "Sample Video":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            videos_dir = os.path.join(base_dir, "videos")
            data_dir = os.path.join(base_dir, "data")
            
            preferred_keywords = ["crowd", "person", "people"] if config_name == "person" else ["traffic", "car", "vehicle"]
            
            sample_videos = {}
            auto_selected = None
            
            if os.path.exists(videos_dir):
                for f in os.listdir(videos_dir):
                    if f.endswith('.mp4'):
                        display_name = f.replace('.mp4', '').title()
                        sample_videos[display_name] = os.path.join(videos_dir, f)
                        if auto_selected is None:
                            for kw in preferred_keywords:
                                if kw.lower() in f.lower():
                                    auto_selected = display_name
                                    break
            
            legacy_path = os.path.join(data_dir, "sample_video.mp4")
            if os.path.exists(legacy_path):
                sample_videos["Default Sample"] = legacy_path
                if auto_selected is None:
                    auto_selected = "Default Sample"
            
            if sample_videos:
                video_list = list(sample_videos.keys())
                default_idx = video_list.index(auto_selected) if auto_selected in video_list else 0
                selected = st.selectbox("Choose Video", video_list, index=default_idx)
                video_path = sample_videos[selected]
            else:
                st.error("No sample videos found")
                
        elif video_source_type == "Upload Video":
            uploaded = st.file_uploader("Upload MP4", type=['mp4', 'avi'])
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                video_path = tfile.name
        else:
            use_camera = True
            camera_id = st.number_input("Camera ID", 0, 5, 0)
        
        st.markdown("---")
        
        # Model Config
        model_name = st.selectbox("Model", ["yolo26l.pt", "yolo26n.pt", "yolo26m.pt"], help="Nano=fast, Large=accurate")
        conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.3, help="Detection threshold")
        skip_frames = st.slider("Skip Frames", 1, 5, 1, help="Process every Nth frame")
        
        st.markdown("---")
        
        # Zone Analysis
        enable_zone = st.checkbox("Enable Zone Occupancy", help="Track objects in specific area")
        zone_params = None
        
        if enable_zone:
            st.markdown("###### Zone Configuration (Normalized 0.0-1.0)")
            z_x = st.slider("Zone X", 0.0, 1.0, 0.2, 0.05)
            z_y = st.slider("Zone Y", 0.0, 1.0, 0.2, 0.05)
            z_w = st.slider("Zone Width", 0.1, 1.0, 0.6, 0.05)
            z_h = st.slider("Zone Height", 0.1, 1.0, 0.6, 0.05)
            zone_params = (z_x, z_y, z_w, z_h)
    
    # Main Area
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        st.markdown('<h3><i class="fa-solid fa-video" style="margin-right:10px; color:var(--accent);"></i>Live Feed</h3>', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        
        # Dynamic buttons
        if not st.session_state.get('processing', False):
            if st.button("▶ Start Detection"):
                st.session_state.processing = True
                st.session_state.processor_initialized = False
                st.session_state.final_counts = None
                st.rerun()
        else:
            if st.button("⏹ Stop Detection"):
                if 'processor' in st.session_state:
                    st.session_state.final_counts = dict(st.session_state.processor.cumulative_counts)
                st.session_state.processing = False
                st.rerun()
    
    with col_stats:
        st.markdown('<h3><i class="fa-solid fa-chart-column" style="margin-right:10px; color:var(--accent);"></i>Analytics</h3>', unsafe_allow_html=True)
        stats_container = st.empty()
        chart_container = st.empty()
    
    # Processing Logic
    if st.session_state.get('processing', False):
        # Initialize processor
        if not st.session_state.get('processor_initialized', False):
            with st.spinner("Initializing YOLO26 model..."):
                processor = YOLOProcessor(
                    model_name=model_name,
                    config_name=config_name,
                    confidence=conf_threshold
                )
                processor.reset_counts()
                st.session_state.processor = processor
                st.session_state.processor_initialized = True
        else:
            processor = st.session_state.processor
        
        # Open video
        if st.session_state.get('cap') is None:
            if use_camera:
                cap = cv2.VideoCapture(camera_id)
            elif video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
            else:
                st.error("No valid video source")
                st.session_state.processing = False
                cap = None
            st.session_state.cap = cap
        else:
            cap = st.session_state.cap
        
        if cap and cap.isOpened():
            frame_idx = 0
            
            while cap.isOpened() and st.session_state.get('processing', False):
                ret, frame = cap.read()
                
                if not ret:
                    if not use_camera:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                frame_idx += 1
                if frame_idx % skip_frames != 0:
                    continue
                
                # Update zone if enabled
                if enable_zone and zone_params:
                    # Calculate absolute coordinates
                    h, w = frame.shape[:2]
                    zx, zy, zw, zh = zone_params
                    
                    # Clamp values
                    x1 = int(max(0, min(zx * w, w - 1)))
                    y1 = int(max(0, min(zy * h, h - 1)))
                    x2 = int(max(x1 + 1, min((zx + zw) * w, w)))
                    y2 = int(max(y1 + 1, min((zy + zh) * h, h)))
                    
                    # Define polygon (rectangle)
                    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    
                    # Update processor zone
                    processor.set_zone(polygon, (w, h))
                elif hasattr(processor, 'zone') and not enable_zone:
                    # Clear zone if disabled
                    processor.zone = None
                
                annotated, counts = processor.process_frame(frame)
                frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
                
                with stats_container.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{counts['total']}</div>
                        <div class="metric-label">Total Unique Objects</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'zone_occupancy' in counts:
                         st.markdown(f"""
                        <div class="metric-card" style="border-color: var(--secondary);">
                            <div class="metric-value" style="color: var(--secondary);">{counts['zone_occupancy']}</div>
                            <div class="metric-label">Zone Occupancy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for cls_name, count in counts.items():
                        if cls_name != 'total' and count > 0:
                            st.metric(cls_name.title(), count)
                
                with chart_container.container():
                    chart = create_donut_chart(counts)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"chart_{frame_idx % 1000}")
            
            cap.release()
            st.session_state.cap = None
            st.session_state.processing = False
    
    # Display final counts
    elif st.session_state.get('final_counts'):
        with stats_container.container():
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.final_counts['total']}</div>
                <div class="metric-label">Final Total</div>
            </div>
            """, unsafe_allow_html=True)
            
            for cls_name, count in st.session_state.final_counts.items():
                if cls_name != 'total' and count > 0:
                    st.metric(cls_name.title(), count)
        
        with chart_container.container():
            chart = create_donut_chart(st.session_state.final_counts)
            if chart:
                st.plotly_chart(chart, use_container_width=True, key="final_chart")

# ==================== MAIN APP ====================
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "landing"
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'final_counts' not in st.session_state:
    st.session_state.final_counts = None

# Inject global CSS
inject_global_css()

# Page Router
if st.session_state.page == "landing":
    render_landing_page()
else:
    render_dashboard()

