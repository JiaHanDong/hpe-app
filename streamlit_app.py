import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 设置页面配置
st.set_page_config(
    page_title="YOLO11 人体姿态估计 (实时流版)",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全局设置和模型加载 ---

# 加载 YOLO11 姿态模型
@st.cache_resource
def load_model():
    """加载 YOLO11 姿态模型"""
    try:
        model = YOLO('yolo11n-pose.pt')
        st.success("✅ YOLO11 姿态模型已在云端成功加载！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

model = load_model()

# --- 核心处理逻辑 (用于 WebRTC) ---

class PoseDetectionTransformer(VideoTransformerBase):
    """
    自定义的视频流处理器。
    每收到一帧视频，就会调用 transform 方法进行处理。
    """
    def __init__(self):
        self.conf_threshold = 0.5 # 默认置信度

    def set_conf_threshold(self, conf):
        """更新置信度阈值"""
        self.conf_threshold = conf

    def transform(self, frame):
        """
        处理单帧图像的核心方法。
        frame: 输入的视频帧 (av.VideoFrame 对象)
        返回: 处理后的视频帧 (av.VideoFrame 对象)
        """
        if model is None:
            return frame

        # 1. 将 av.VideoFrame 转换为 numpy 数组 (BGR 格式)
        img = frame.to_ndarray(format="bgr24")

        # 2. 使用 YOLO 模型进行姿态估计
        results = model(img, conf=self.conf_threshold)

        # 3. 在原始图像上绘制检测结果
        annotated_img = results[0].plot()

        # 4. 将处理后的 numpy 数组转换回 av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- UI 布局 ---

# 侧边栏
with st.sidebar:
    st.title('👤 YOLO11 人体姿态估计')
    st.markdown("""
    这个应用在**云端**运行，使用**您本地电脑的摄像头**进行**实时**人体姿态估计。
    请选择输入源开始。
    """)
    st.divider()

    source = st.selectbox("请选择输入源", ["图片", "视频", "摄像头 (实时流)"])
    
    # 初始化 session_state
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.5
    
    # 置信度滑块
    conf_threshold = st.slider("检测置信度", 0.0, 1.0, st.session_state.conf_threshold, 0.05)
    
    st.divider()
    st.markdown("© 2024 Streamlit & YOLO11")

# 主页面
st.title("YOLO11 人体姿态估计演示 (实时流版)")

if model is not None:
    if source == "图片":
        st.subheader("上传一张图片进行姿态估计")
        uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 原始图片")
                st.image(uploaded_file, use_column_width=True)
            with col2:
                st.markdown("### 姿态估计结果")
                if st.button("开始检测"):
                    with st.spinner("正在云端检测姿态..."):
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
                        results = model(img, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    elif source == "视频":
        st.subheader("上传一个视频文件进行姿态估计")
        uploaded_file = st.file_uploader("选择一个视频", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("❌ 无法打开上传的视频文件。")
            else:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.write(f"📊 视频属性: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")
                
                if st.button("开始在云端处理视频"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    output_filename = "output_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                    
                    with st.spinner("🎬 正在云端处理视频，这可能需要一些时间..."):
                        for frame_count in range(total_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            results = model(frame, conf=conf_threshold)
                            annotated_frame = results[0].plot()
                            out.write(annotated_frame)
                            
                            progress = (frame_count + 1) / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"处理进度: {frame_count + 1}/{total_frames} ({progress:.1%})")
                    
                    cap.release()
                    out.release()
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ 视频处理完成！")
                    
                    with open(output_filename, 'rb') as f:
                        st.download_button(
                            label='📥 下载处理后的视频',
                            data=f,
                            file_name=output_filename,
                            mime='video/mp4'
                        )

    elif source == "摄像头 (实时流)":
        st.subheader("使用您的本地摄像头进行实时姿态估计")
        st.markdown("""
        请允许浏览器访问您的摄像头。应用会将您的摄像头画面**实时**传输到云端进行处理，
        并将结果**实时**返回显示。您可以通过侧边栏的滑块实时调整检测置信度。
        """)
        
        # 创建 PoseDetectionTransformer 的实例
        pose_transformer = PoseDetectionTransformer()
        # 初始设置置信度
        pose_transformer.set_conf_threshold(conf_threshold)

        # 配置 WebRTC
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        # 使用 webrtc_streamer 组件启动实时流
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            video_transformer_factory=lambda: pose_transformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        # 关键修复：使用一个空的占位符来动态更新置信度
        # 这个占位符不会显示任何内容，但会在每次置信度变化时触发UI更新
        status_placeholder = st.empty()
        
        # 如果 WebRTC 连接已建立
        if webrtc_ctx.state.playing:
            # 持续检查置信度滑块是否有变化
            if st.session_state.conf_threshold != conf_threshold:
                # 更新处理器中的置信度
                pose_transformer.set_conf_threshold(conf_threshold)
                # 更新 session_state 以避免重复触发
                st.session_state.conf_threshold = conf_threshold
                
                # 在占位符中短暂显示一个更新提示，然后立即清空
                with status_placeholder:
                    st.success("置信度已更新！")
                    time.sleep(1)
                status_placeholder.empty()

else:
    st.error("由于模型加载失败，应用无法正常运行。请检查云端环境的网络连接和配置。")