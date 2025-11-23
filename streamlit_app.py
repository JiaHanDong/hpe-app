import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import time

# --- 全局设置和补丁 ---

# 1. 解决 OpenCV "The function is not implemented" 错误
# 将 cv2.destroyAllWindows 替换为一个空操作，因为我们不需要它
original_destroy_all_windows = cv2.destroyAllWindows
cv2.destroyAllWindows = lambda: None

# 设置页面配置
st.set_page_config(
    page_title="YOLO11 人体姿态估计",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 辅助函数 ---

def get_available_cameras(max_checks=10):
    """
    检测并返回系统上所有可用的摄像头索引列表。
    """
    available_cameras = []
    for i in range(max_checks):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        # 短暂延迟，防止某些摄像头初始化过慢
        time.sleep(0.05)
    return available_cameras

# --- 模型加载 ---

@st.cache_resource
def load_model():
    """加载 YOLO11 姿态模型 (使用轻量级的 n 版本以提高速度)"""
    try:
        model = YOLO('yolo11n-pose.pt')
        st.success("✅ YOLO11 姿态模型已成功加载！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

# --- UI 布局 ---

# 侧边栏
with st.sidebar:
    st.title('👤 YOLO11 人体姿态估计')
    st.markdown("""
    这个应用使用 Ultralytics YOLO11 模型进行实时人体姿态估计。
    支持图片、视频上传和摄像头实时检测。
    """)
    st.divider()

    # 1. 选择输入源
    source = st.selectbox("请选择输入源", ["图片", "视频", "摄像头"])
    
    # 2. 根据输入源显示不同的配置
    conf_threshold = st.slider("检测置信度", 0.0, 1.0, 0.5, 0.05)
    
    camera_index = 0
    if source == "摄像头":
        available_cameras = get_available_cameras()
        if not available_cameras:
            st.warning("⚠️ 未检测到可用的摄像头。请确保摄像头已正确连接。")
        else:
            camera_index = st.selectbox("选择摄像头", available_cameras)
    
    st.divider()
    st.markdown("© 2024 Streamlit & YOLO11")

# 主页面
st.title("YOLO11 人体姿态估计演示")
model = load_model()

# --- 核心逻辑处理 ---

if model is not None:
    if source == "图片":
        st.subheader("上传一张图片进行姿态估计")
        uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            # 显示原始图片
            with col1:
                st.markdown("### 原始图片")
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # 检测并显示结果
            with col2:
                st.markdown("### 姿态估计结果")
                if st.button("开始检测"):
                    with st.spinner("正在检测姿态..."):
                        results = model(img, conf=conf_threshold)
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
                
                if st.button("开始处理视频"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    output_filename = "output_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                    
                    with st.spinner("🎬 正在处理视频，这可能需要一些时间..."):
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

    elif source == "摄像头":
        st.subheader("使用摄像头进行实时姿态估计")
        
        # 创建两个列，一个用于显示画面，一个用于控制
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 控制")
            start_button = st.button("▶️ 开始")
            stop_button = st.button("⏹️ 停止")
        
        with col1:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
        
        is_running = False
        cap = None
        
        if start_button:
            is_running = True
            # 使用用户选择的摄像头索引
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error(f"❌ 无法打开摄像头 (索引: {camera_index})。请检查摄像头是否被其他程序占用或尝试其他索引。")
                is_running = False
            
            # 设置摄像头分辨率（可选，根据你的硬件调整）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while is_running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("⚠️ 无法读取摄像头画面。")
                break
            
            # 进行姿态检测
            results = model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            
            # 显示画面
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            status_placeholder.markdown("🟢 实时检测中...")
            
            # 检查是否按下停止按钮
            if stop_button:
                is_running = False
                break
        
        if cap is not None:
            cap.release()
        
        if stop_button or not is_running:
            frame_placeholder.empty()
            status_placeholder.markdown("⏹️ 检测已停止。")

else:
    st.error("由于模型加载失败，应用无法正常运行。请检查你的网络连接和环境配置。")