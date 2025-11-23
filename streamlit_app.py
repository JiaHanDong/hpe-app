import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# 设置页面配置
st.set_page_config(
    page_title="YOLO11 人体姿态估计 (云端版)",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全局设置和模型加载 ---

# 加载 YOLO11 姿态模型 (使用轻量级的 n 版本以提高速度)
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

# --- 核心处理函数 ---

def process_camera_frame():
    """
    当摄像头捕获到新图像时触发的回调函数。
    该函数在云端服务器上执行。
    """
    if 'camera' not in st.session_state or st.session_state.camera is None:
        return

    # 1. 获取从用户浏览器传来的图像数据
    camera_image = st.session_state.camera

    # 2. 将图像数据转换为 OpenCV 格式
    # camera_image 是一个 UploadedFile 对象，我们需要先读取其字节
    img_bytes = camera_image.getvalue()
    # 用 numpy 把字节转换成数组，再用 cv2.imdecode 解码成图像
    # cv2.IMREAD_COLOR 会忽略透明度通道，返回 BGR 格式
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.warning("⚠️ 无法解码摄像头图像。")
        return

    # 3. 在云端使用 YOLO 模型进行姿态估计
    if model:
        results = model(frame, conf=st.session_state.conf_threshold)
        # 绘制结果
        annotated_frame = results[0].plot()

        # 4. 将处理后的 OpenCV 图像转换回 PIL Image 格式，以便在前端显示
        # OpenCV 图像是 BGR 格式，需要先转换为 RGB
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # 创建一个 BytesIO 缓冲区
        buf = io.BytesIO()
        # 将 RGB 图像保存到缓冲区，格式为 PNG
        Image.fromarray(rgb_frame).save(buf, format="PNG")
        # 将缓冲区的指针移到开头
        buf.seek(0)

        # 5. 将处理后的图像存储在 session_state 中，供主界面显示
        st.session_state.processed_frame = buf

# --- UI 布局 ---

# 侧边栏
with st.sidebar:
    st.title('👤 YOLO11 人体姿态估计')
    st.markdown("""
    这个应用在**云端**运行，但会使用**您本地电脑的摄像头**进行实时姿态估计。
    请选择输入源开始。
    """)
    st.divider()

    # 1. 选择输入源
    source = st.selectbox("请选择输入源", ["图片", "视频", "摄像头 (本地)"])
    
    # 2. 检测置信度滑块
    st.session_state.conf_threshold = st.slider("检测置信度", 0.0, 1.0, 0.5, 0.05)
    
    st.divider()
    st.markdown("© 2024 Streamlit & YOLO11")

# 主页面
st.title("YOLO11 人体姿态估计演示 (云端版)")

# 初始化 session_state，用于在回调函数和主程序之间共享数据
if 'processed_frame' not in st.session_state:
    st.session_state.processed_frame = None

if model is not None:
    if source == "图片":
        st.subheader("上传一张图片进行姿态估计")
        uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 原始图片")
                # 直接显示上传的图片
                st.image(uploaded_file, use_column_width=True)
            
            with col2:
                st.markdown("### 姿态估计结果")
                if st.button("开始检测"):
                    with st.spinner("正在云端检测姿态..."):
                        # 处理逻辑与之前类似
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
                        results = model(img, conf=st.session_state.conf_threshold)
                        annotated_img = results[0].plot()
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    elif source == "视频":
        st.subheader("上传一个视频文件进行姿态估计")
        uploaded_file = st.file_uploader("选择一个视频", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_file is not None:
            # 保存上传的视频到临时文件
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
                                
                            results = model(frame, conf=st.session_state.conf_threshold)
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

    elif source == "摄像头 (本地)":
        st.subheader("使用您的本地摄像头进行实时姿态估计")
        st.markdown("""
        请允许浏览器访问您的摄像头。应用会将您的摄像头画面实时传输到云端进行处理，
        并将结果返回显示。由于网络传输和云端处理，可能会有轻微延迟。
        """)
        
        # 创建一个占位符来显示处理后的实时画面
        frame_placeholder = st.empty()
        
        # 使用 st.camera_input 来在用户浏览器中启动摄像头
        # key 是必需的，用于触发 on_change 事件
        # on_change 绑定到我们的处理函数 process_camera_frame
        camera_input = st.camera_input(
            "请面对摄像头",
            key="camera",
            on_change=process_camera_frame
        )

        # 主循环：持续检查 session_state 中是否有处理好的帧，如果有则显示
        while True:
            if st.session_state.processed_frame is not None:
                # 将处理后的帧显示在占位符中
                frame_placeholder.image(st.session_state.processed_frame, channels="RGB", use_column_width=True)
            
            # 添加一个小延迟，降低CPU占用
            st.sleep(0.01)

else:
    st.error("由于模型加载失败，应用无法正常运行。请检查云端环境的网络连接和配置。")