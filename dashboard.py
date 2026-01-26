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
import pandas as pd
import atexit
from typing import Tuple, Dict, Optional, List
from vehicle_counter import YOLOProcessor
from config import (
    DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, TEAL_BGR,
    DEFAULT_CONFIDENCE, AVAILABLE_MODELS, DEFAULT_MODEL,
    PREVIEW_TEXT, WEBCAM_PREVIEW_TEXT, ERROR_NO_VIDEO, ERROR_NO_SAMPLES,
    CHART_UPDATE_INTERVAL
)

# Track temp files for cleanup
_temp_files: List[str] = []

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

# Page Config
st.set_page_config(
    page_title="YOLO Vision AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS - Organized in external file, loaded properly
@st.cache_data
def load_css_content() -> str:
    """Cache CSS content to avoid re-reading from disk on every rerun"""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""  # Fallback if file missing

def inject_global_css() -> None:
    """Load CSS from external file for better maintainability"""
    css_content = load_css_content()  # Now cached!
    
    # Inject Font Awesome and Google Fonts
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    # Inject CSS
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
def create_donut_chart(counts) -> go.Figure:
    """Create class distribution donut chart from DetectionCount"""
    if not counts:
        return None
    
    # Handle DetectionCount dataclass
    counts_dict = counts.counts_by_class if hasattr(counts, 'counts_by_class') else counts
    total = counts.total if hasattr(counts, 'total') else counts.get('total', 0)
    
    if not counts_dict or len(counts_dict) < 1:
        return None
    
    labels = []
    values = []
    for key, val in counts_dict.items():
        if val > 0:
            labels.append(key.title())
            values.append(val)
    
    if not labels:
        return None
    
    # Premium color palette matching neo-noir theme
    premium_colors = [
        '#00D4FF',  # Electric cyan
        '#7C3AED',  # Purple
        '#F472B6',  # Pink
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#3B82F6',  # Blue
        '#8B5CF6',  # Violet
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(
            colors=premium_colors[:len(labels)],
            line=dict(color='rgba(26, 26, 46, 0.8)', width=3)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, color='#E2E8F0', family='Inter'),
        hoverinfo='label+value+percent',
        hoverlabel=dict(
            bgcolor='rgba(26, 26, 46, 0.95)',
            font_size=14,
            font_color='#00D4FF'
        ),
        pull=[0.02] * len(labels),  # Slight pull for 3D effect
    )])
    
    # Center annotation with total count
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=30),
        height=280,
        annotations=[
            dict(
                text=f'<b style="font-size:28px; color:#00D4FF">{total}</b><br><span style="font-size:11px; color:#94A3B8">DETECTED</span>',
                x=0.5, y=0.5,
                font=dict(size=14, color='#E2E8F0', family='Inter'),
                showarrow=False,
                align='center'
            )
        ]
    )
    return fig

def get_preview_frame(video_source_type: str, video_path: str = None, camera_id: int = 0,
                      line_params: dict = None, zone_params: tuple = None):
    """
    Get a preview frame based on the selected video source.
    - Sample Video: First frame in grayscale with config overlay
    - Upload Video: Black screen
    - Webcam: Live frame from camera with config overlay
    Returns: (frame, is_live) - frame is BGR numpy array, is_live indicates if it's a live feed
    """
    default_height, default_width = DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
    frame = None
    is_live = False
    
    if video_source_type == "Sample Video" and video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, raw_frame = cap.read()
        cap.release()
        if ret:
            # Convert to grayscale and back to BGR for display
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif video_source_type == "Upload Video":
        # Black screen with text
        frame = np.zeros((default_height, default_width, 3), dtype=np.uint8)
        cv2.putText(frame, "Upload a video to begin", (default_width//2 - 180, default_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return frame, False
    
    elif video_source_type == "Webcam":
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()
        cap.release()
        if ret:
            is_live = True
    
    # Fallback: black screen
    if frame is None:
        frame = np.zeros((default_height, default_width, 3), dtype=np.uint8)
        return frame, False
    
    h, w = frame.shape[:2]
    
    # Draw line configuration preview
    if line_params:
        line_y_px = int(line_params["y"] * h)
        margin_px = int(line_params["margin"] * w)
        # Draw the counting line (teal color)
        cv2.line(frame, (margin_px, line_y_px), (w - margin_px, line_y_px), 
                (212, 245, 0), 2)  # Teal in BGR
        # Draw circles at ends
        cv2.circle(frame, (margin_px, line_y_px), 6, (212, 245, 0), -1)
        cv2.circle(frame, (w - margin_px, line_y_px), 6, (212, 245, 0), -1)
    
    # Draw zone configuration preview
    if zone_params:
        zx, zy, zw, zh = zone_params
        x1, y1 = int(zx * w), int(zy * h)
        x2, y2 = int((zx + zw) * w), int((zy + zh) * h)
        # Draw zone rectangle with teal color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (212, 245, 0), 2)
        # Add zone label
        cv2.putText(frame, "ZONE", (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (212, 245, 0), 2)
    
    # Add preview text
    mode_text = PREVIEW_TEXT
    if is_live:
        mode_text = WEBCAM_PREVIEW_TEXT
    cv2.putText(frame, mode_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    return frame, is_live


def render_dashboard():
    """Render the detection dashboard"""
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "landing"
        st.session_state.processing = False
        st.rerun()
    
    # Title
    st.markdown('<h1 style="text-align: center;"><i class="fa-solid fa-eye" style="color:var(--electric); margin-right:15px;"></i>YOLO Vision <span style="color:var(--accent);">AI</span></h1>', unsafe_allow_html=True)
    
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
                # Use a key tied to config_name to force reset when detection mode changes
                selected = st.selectbox("Choose Video", video_list, index=default_idx, key=f"video_select_{config_name}")
                video_path = sample_videos[selected]
            else:
                st.error("No sample videos found")
                
        elif video_source_type == "Upload Video":
            uploaded = st.file_uploader("Upload MP4", type=['mp4', 'avi'])
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                video_path = tfile.name
                _temp_files.append(video_path)  # Track for cleanup
        else:
            use_camera = True
            camera_id = st.number_input("Camera ID", min_value=0, max_value=5, value=0)
        
        st.markdown("---")
        
        # Model Config
        model_name = st.selectbox("Model", AVAILABLE_MODELS, help="YOLOv8n = fastest but less accurate. YOLOv8l/YOLO26l = slower but most accurate. Choose based on your hardware.")
        conf_threshold = st.slider("Confidence", 0.1, 1.0, DEFAULT_CONFIDENCE, help="Minimum confidence to detect an object. Low (0.1) = detect more objects but more false positives. High (1.0) = only very confident detections.")
        skip_frames = st.slider("Skip Frames", min_value=1, max_value=5, value=1, help="Process every Nth frame. 1 = process all frames (accurate but slow). 5 = process every 5th frame (fast but may miss quick movements).")
        
        st.markdown("---")
        
        # Analysis Mode
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Line Crossing", "Area Density"],
            help="Line: count objects crossing a line. Area: count objects in a zone."
        )
        
        # Mode-specific settings
        line_params = None
        zone_params = None
        enable_zone = False
        
        if analysis_mode == "Line Crossing":
            st.markdown("###### Line Configuration")
            st.info("Objects crossing this line will be counted")
            line_y = st.slider("Line Y Position", 0.0, 1.0, 0.5, 0.05, help="Where to place the counting line vertically. 0.0 = top of video, 0.5 = middle, 1.0 = bottom. Place where vehicles cross the screen.")
            line_margin = st.slider("Margin", 0.0, 0.5, 0.0, 0.05, help="Shorten the line from both edges. 0.0 = line spans full width, 0.5 = line is only in the center. Useful to ignore objects at screen edges.")
            line_params = {"y": line_y, "margin": line_margin}
        else:  # Area Density
            st.markdown("###### Zone Configuration")
            st.info("Objects inside this zone will be counted")
            enable_zone = True
            z_x = st.slider("Zone X", 0.0, 1.0, 0.2, 0.05, help="Left edge of the zone. 0.0 = left side of video, 1.0 = right side.")
            z_y = st.slider("Zone Y", 0.0, 1.0, 0.2, 0.05, help="Top edge of the zone. 0.0 = top of video, 1.0 = bottom.")
            z_w = st.slider("Zone Width", 0.1, 1.0, 0.6, 0.05, help="How wide the zone is. 0.1 = small (10% of video width), 1.0 = full width.")
            z_h = st.slider("Zone Height", 0.1, 1.0, 0.6, 0.05, help="How tall the zone is. 0.1 = small (10% of video height), 1.0 = full height.")
            zone_params = (z_x, z_y, z_w, z_h)
    
    # Main Area
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        st.markdown('<h3><i class="fa-solid fa-video" style="margin-right:10px; color:var(--accent);"></i>Live Feed</h3>', unsafe_allow_html=True)
        
        frame_placeholder = st.empty()
        
        # Show preview frame when not processing
        if not st.session_state.get('processing', False):
            preview_frame, is_live = get_preview_frame(
                video_source_type, video_path, camera_id,
                line_params=line_params, zone_params=zone_params
            )
            frame_placeholder.image(preview_frame, channels="BGR", width="stretch")
        
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
                    st.session_state.final_counts = st.session_state.processor.current_counts
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
        
        # Open video - detect if video source changed
        current_source = video_path if video_path else f"camera_{camera_id}"
        previous_source = st.session_state.get('current_video_source')
        
        # If source changed or no cap exists, open new video
        if st.session_state.get('cap') is None or previous_source != current_source:
            # Release old capture if exists
            if st.session_state.get('cap') is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            
            if use_camera:
                cap = cv2.VideoCapture(camera_id)
            elif video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
            else:
                st.error("No valid video source")
                st.session_state.processing = False
                cap = None
            st.session_state.cap = cap
            st.session_state.current_video_source = current_source
        else:
            cap = st.session_state.cap
        
        
        if cap and cap.isOpened():
            # Set up line ONCE before processing (if in Line Crossing mode)
            if line_params:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                line_y_px = int(line_params["y"] * h)
                margin_px = int(line_params["margin"] * w)
                # Set line for crossing detection (only once!)
                processor.set_line(
                    start=(margin_px, line_y_px),
                    end=(w - margin_px, line_y_px)
                )
            
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_camera else 0
            progress_bar = None
            if total_frames > 0:
                progress_bar = st.progress(0, text="Processing video...")
            
            while cap.isOpened() and st.session_state.get('processing', False):
                ret, frame = cap.read()
                
                if not ret:
                    # Video ended - stop processing
                    break
                
                frame_idx += 1
                if frame_idx % skip_frames != 0:
                    continue
                
                
                # Update zone if enabled (Area Density mode)
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
                elif line_params:
                    # Line Crossing mode - clear zone but keep line
                    if hasattr(processor, 'region_zone'):
                        processor.region_zone = None
                else:
                    # Clear both line and zone if neither enabled
                    if hasattr(processor, 'line_zone'):
                        processor.line_zone = None
                    if hasattr(processor, 'region_zone'): # Changed from 'zone' to 'region_zone' for consistency
                        processor.region_zone = None
                
                annotated, counts = processor.process_frame(frame)
                
                # Draw counting line for Line Crossing mode
                if line_params:
                    h, w = annotated.shape[:2]
                    line_y_px = int(line_params["y"] * h)
                    margin_px = int(line_params["margin"] * w)
                    # Draw the counting line (teal color)
                    cv2.line(annotated, (margin_px, line_y_px), (w - margin_px, line_y_px), 
                            (212, 245, 0), 2)  # Teal in BGR
                    # Draw small markers at ends
                    cv2.circle(annotated, (margin_px, line_y_px), 6, (212, 245, 0), -1)
                    cv2.circle(annotated, (w - margin_px, line_y_px), 6, (212, 245, 0), -1)
                
                frame_placeholder.image(annotated, channels="BGR", width="stretch")
                
                # Store last frame for snapshot feature
                st.session_state.last_frame = annotated.copy()
                
                with stats_container.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{counts.total}</div>
                        <div class="metric-label">Total Unique Objects</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show direction counters for Line Crossing mode
                    if line_params and hasattr(counts, 'in_count'):
                        col_in, col_out = st.columns(2)
                        with col_in:
                            st.metric("↓ In", counts.in_count)
                        with col_out:
                            st.metric("↑ Out", counts.out_count)
                    
                    if hasattr(counts, 'zone_occupancy') and counts.zone_occupancy > 0:
                         st.markdown(f"""
                        <div class="metric-card" style="border-color: var(--secondary);">
                            <div class="metric-value" style="color: var(--secondary);">{counts.zone_occupancy}</div>
                            <div class="metric-label">Zone Occupancy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for cls_name, count in counts.counts_by_class.items():
                        if count > 0:
                            st.metric(cls_name.title(), count)
                
                with chart_container.container():
                    # Only update chart every N frames to reduce flickering
                    if frame_idx % CHART_UPDATE_INTERVAL == 0:
                        chart = create_donut_chart(counts)
                        if chart:
                            st.plotly_chart(chart, width="stretch", key=f"chart_{frame_idx % 1000}")
                
                # Update progress bar
                if progress_bar and total_frames > 0:
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress, text=f"Processing: {int(progress * 100)}%")
            
            cap.release()
            st.session_state.cap = None
            st.session_state.processing = False
    
    # Display final counts
    elif st.session_state.get('final_counts'):
        final = st.session_state.final_counts
        with stats_container.container():
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{final.total}</div>
                <div class="metric-label">Final Total</div>
            </div>
            """, unsafe_allow_html=True)
            
            for cls_name, count in final.counts_by_class.items():
                if count > 0:
                    st.metric(cls_name.title(), count)
            
            # ===== CSV EXPORT =====
            st.markdown("---")
            st.markdown("##### 📥 Export Data")
            
            # Prepare CSV data
            csv_data = []
            for cls_name, count in final.counts_by_class.items():
                csv_data.append({"Class": cls_name.title(), "Count": count})
            
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV",
                data=csv_string,
                file_name="detection_counts.csv",
                mime="text/csv",
                help="Download count data as CSV file"
            )
        
        # ===== SNAPSHOT BUTTON =====
        if st.session_state.get('last_frame') is not None:
            with col_video:
                st.markdown("##### 📸 Snapshot")
                # Encode frame to PNG bytes
                _, buffer = cv2.imencode('.png', st.session_state.last_frame)
                st.download_button(
                    label="📷 Save Snapshot",
                    data=buffer.tobytes(),
                    file_name="detection_snapshot.png",
                    mime="image/png",
                    help="Save the last processed frame as an image"
                )
        
        with chart_container.container():
            chart = create_donut_chart(st.session_state.final_counts)
            if chart:
                st.plotly_chart(chart, width="stretch", key="final_chart")

# ==================== MAIN APP ====================
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "landing"
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'final_counts' not in st.session_state:
    st.session_state.final_counts = None
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# Inject global CSS
inject_global_css()

# Page Router
if st.session_state.page == "landing":
    render_landing_page()
else:
    render_dashboard()

