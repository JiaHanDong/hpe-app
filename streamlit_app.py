import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="YOLO11 人体姿态估计",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 在侧边栏添加标题和说明
with st.sidebar:
    st.title('👤 YOLO11 人体姿态估计')
    st.markdown("""
    这个应用使用 Ultralytics YOLO11 模型来进行实时人体姿态估计。
    您可以通过以下方式上传内容：
    - 上传一张图片
    - 上传一个视频文件
    - 使用您的摄像头进行实时检测
    """)
    
    # 选择输入源
    source = st.selectbox("选择输入源", ["图片", "视频", "摄像头"])
    
    # 置信度滑块
    confidence = st.slider("检测置信度", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.markdown("© 2024 Streamlit & YOLO11")

# 主页面标题
st.title("YOLO11 人体姿态估计演示")

# 加载 YOLO11 姿态模型
@st.cache_resource
def load_model():
    """加载 YOLO11 姿态模型"""
    model = YOLO('yolo11n-pose.pt')
    return model

model = load_model()
st.success("YOLO11 姿态模型已成功加载！")

# 根据选择的输入源进行处理
if source == "图片":
    st.subheader("上传一张图片进行姿态估计")
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 将上传的文件转换为 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 在页面上显示原始图片
        st.subheader("原始图片")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # 进行姿态检测
        if st.button("开始检测"):
            with st.spinner("正在检测姿态..."):
                results = model(img, conf=confidence)
                
                # 绘制检测结果
                annotated_img = results[0].plot()
                
                # 显示检测结果
                st.subheader("姿态估计结果")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

elif source == "视频":
    st.subheader("上传一个视频文件进行姿态估计")
    uploaded_file = st.file_uploader("选择一个视频", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # 打开视频文件
        cap = cv2.VideoCapture(tfile.name)
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.write(f"视频属性: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")
        
        # 创建视频写入器
        output_filename = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        # 处理视频
        if st.button("开始处理视频"):
            progress_bar = st.progress(0)
            frame_count = 0
            
            with st.spinner("正在处理视频，这可能需要一些时间..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 进行姿态检测
                    results = model(frame, conf=confidence)
                    annotated_frame = results[0].plot()
                    
                    # 写入处理后的帧
                    out.write(annotated_frame)
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
            
            # 释放资源
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            st.success("视频处理完成！")
            
            # 提供下载链接
            with open(output_filename, 'rb') as f:
                st.download_button('下载处理后的视频', f, file_name=output_filename)

elif source == "摄像头":
    st.subheader("使用摄像头进行实时姿态估计")
    
    # 创建一个占位符来显示摄像头画面
    frame_placeholder = st.empty()
    
    # 创建一个停止按钮
    stop_button_pressed = st.button("停止")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取摄像头画面")
            break
        
        # 进行姿态检测
        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()
        
        # 显示画面
        frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()
    st.write("摄像头已停止")