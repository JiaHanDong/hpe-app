import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import av
import time
# 导入新版所需的基类和配置
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# 设置页面配置
st.set_page_config(
    page_title="人体姿态估计演示",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 模型加载 ---
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolo11n-pose.pt')
        st.success("✅ 姿态模型已加载！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

model = load_model()

# --- 核心视频处理器（适配新版 API）---
class PoseDetectionProcessor(VideoProcessorBase):  # 注意：使用新版 VideoProcessorBase
    """实时视频帧处理器，每帧都会被自动调用"""
    def __init__(self):
        self.conf_threshold = 0.5  # 初始置信度

    def set_conf_threshold(self, conf):
        """更新置信度阈值（支持实时调整）"""
        self.conf_threshold = conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:  # 新版用 recv 方法处理帧
        """处理单帧视频并返回结果"""
        if model is None:
            return frame

        # 1. 将视频帧转换为 OpenCV 格式（BGR）
        img = frame.to_ndarray(format="bgr24")

        # 2. 用 YOLO 模型检测姿态
        results = model(img, conf=self.conf_threshold)

        # 3. 绘制检测结果
        annotated_img = results[0].plot()

        # 4. 转换回视频帧格式并返回
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- UI 布局 ---
with st.sidebar:
    st.title('👤 人体姿态估计')

    st.markdown("""
    支持通过上传图片、视频以及使用本地摄像头作为输入，并对输入内容中的人体进行姿态估计。
    可通过滑块调整检测灵敏度。
    """)
    st.markdown("""
    作者：王婷婷
    """)
    st.divider()

    source = st.selectbox("输入源", ["图片", "视频", "摄像头 (实时流)"])
    conf_threshold = st.slider("检测置信度", 0.0, 1.0, 0.5, 0.05)  # 置信度滑块

    st.divider()
    st.caption("© 2025 王婷婷 实时姿态估计演示")

# 主页面
st.title("人体姿态估计演示")

if model is not None:
    if source == "摄像头 (实时流)":
        st.subheader("本地摄像头实时处理")
        st.info("请允许浏览器访问摄像头，画面将实时处理并显示姿态估计结果。")

        # 初始化处理器
        processor = PoseDetectionProcessor()
        processor.set_conf_threshold(conf_threshold)

        # WebRTC 配置（使用谷歌 STUN 服务器穿透防火墙）
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        # 启动实时流（使用新版参数）
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            video_processor_factory=lambda: processor,  # 替换 video_transformer_factory 为 video_processor_factory
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,  # 替换 async_transform 为 async_processing
        )

        # 实时更新置信度（当滑块变化时）
        if webrtc_ctx.state.playing:
            # 持续检查滑块值变化，实时更新处理器参数
            processor.set_conf_threshold(conf_threshold)

    # 图片和视频处理逻辑保持不变（复用之前的稳定代码）
    elif source == "图片":
        st.subheader("图片姿态估计")
        uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="原始图片", use_column_width=True)
            with col2:
                if st.button("开始检测"):
                    with st.spinner("处理中..."):
                        img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, 1)
                        results = model(img, conf=conf_threshold)
                        st.image(
                            cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                            caption="姿态估计结果",
                            use_column_width=True
                        )

    elif source == "视频":
        st.subheader("视频姿态估计")
        uploaded_file = st.file_uploader("选择视频", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("无法打开视频文件")
            else:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width, height = int(cap.get(3)), int(cap.get(4))
                st.write(f"视频信息：{width}x{height}，{fps} FPS")

                if st.button("开始处理"):
                    progress = st.progress(0)
                    status = st.empty()
                    out = cv2.VideoWriter(
                        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (width, height)
                    )

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    with st.spinner("处理中..."):
                        for i in range(total_frames):
                            ret, frame = cap.read()
                            if not ret: break
                            results = model(frame, conf=conf_threshold)
                            out.write(results[0].plot())
                            progress.progress((i+1)/total_frames)
                            status.text(f"处理中：{i+1}/{total_frames}")

                    cap.release()
                    out.release()
                    st.success("处理完成！")
                    with open("output.mp4", 'rb') as f:
                        st.download_button("下载结果", f, "output.mp4")

else:
    st.error("模型加载失败，请检查网络或环境配置。")
