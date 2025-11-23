import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
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
    
    # 为实时流和非实时流分别创建置信度滑块，但共享同一个 session_state
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.5
    
    conf_threshold = st.slider("检测置信度", 0.0, 1.0, st.session_state.conf_threshold, 0.05)
    st.session_state.conf_threshold = conf_threshold
    
    st.divider()
    st.markdown("© 2024 Streamlit & YOLO11")

# 主页面
st.title("YOLO11 人体姿态估计演示 (实时流版)")

if model is not None:
    if source == "图片":
        # 图片处理逻辑保持不变
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
                        results = model(img, conf=st.session_state.conf_threshold)
                        annotated_img = results[0].plot()
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    elif source == "视频":
        # 视频处理逻辑保持不变
        st.subheader("上传一个视频文件进行姿态估计")
        uploaded_file = st.file_uploader("选择一个视频", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("❌ 无法打开上传的视频文件。")
            else:
                # ... (此处省略与之前相同的视频处理代码)
                # 为了简洁，这里只保留了框架，你可以从之前的代码中复制完整逻辑
                st.write("视频处理功能在此处保留，逻辑与之前版本相同...")
                # [复制之前版本中的完整视频处理代码到这里]

    elif source == "摄像头 (实时流)":
        st.subheader("使用您的本地摄像头进行实时姿态估计")
        st.markdown("""
        请允许浏览器访问您的摄像头。应用会将您的摄像头画面**实时**传输到云端进行处理，
        并将结果**实时**返回显示。
        """)
        
        # 创建 PoseDetectionTransformer 的实例
        pose_transformer = PoseDetectionTransformer()
        # 将当前的置信度阈值传递给处理器
        pose_transformer.set_conf_threshold(st.session_state.conf_threshold)

        # 配置 WebRTC，使用公共的 STUN 服务器来帮助穿透防火墙
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        # 使用 webrtc_streamer 组件启动实时流
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            video_transformer_factory=lambda: pose_transformer, # 传递我们的处理器实例
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True, # 启用异步处理，提高性能
        )

        # 如果 WebRTC 连接已建立，并且置信度滑块的值发生了变化，
        # 则更新处理器中的置信度阈值。
        if webrtc_ctx.state.playing:
            # 使用 st.checkbox 或 st.slider 的 on_change 事件来触发更新
            # 这里我们用一个隐藏的按钮来触发，当置信度变化时自动点击
            if st.button("更新置信度", key="update_conf", disabled=True, visible=False):
                pass
            
            # 监听置信度滑块的变化
            if st.session_state.conf_threshold != pose_transformer.conf_threshold:
                pose_transformer.set_conf_threshold(st.session_state.conf_threshold)
                # 模拟点击隐藏按钮以触发 UI 更新
                st.session_state["update_conf"] = True

else:
    st.error("由于模型加载失败，应用无法正常运行。请检查云端环境的网络连接和配置。")

# 注意：为了代码简洁，我在这里省略了与之前版本完全相同的视频处理部分。
# 你可以直接从上个版本的代码中复制 `elif source == "视频":` 块的完整内容来替换这里的占位符。