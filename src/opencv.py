import cv2
import numpy as np
# import wiringpi  # 删除wiringpi导入
import time
import math
import struct
import string
import serial

#  0为输出红色激光与绿色激光的偏差， 1为输出激光点与扫矩形目标点的偏差, 2为输出激光点与一直线往复运动的目标点的偏差，模式3为输出激光点与1目标点的偏差
#  模式2与模式3为调试下位机pid时使用
mode = 0

cap = cv2.VideoCapture(0)
#  打开本地视频调试可在pc端调试时使用
video = cv2.VideoCapture('WIN_20240503_22_01_05_Pro.mp4')
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 120)

def float_to_bytes_big_endian(f_val):

    little_endian = struct.pack('<f', f_val)
    big_endian = little_endian[::-1]  # 反转字节序
    return big_endian

def bytes_to_float_big_endian(byte_data):

    # 先反转为小端序，然后解包
    little_endian = byte_data[::-1]
    return struct.unpack('<f', little_endian)[0]

def create_target_angle_packet(target_yaw, target_pitch):
    packet = bytearray()
    # 包头：固定标识符 [0xAA, 0x55] - 仅用于识别数据包边界
    packet.extend([0xAA, 0x55])
    
   
    packet.extend(float_to_bytes_big_endian(target_yaw))
    
   
    packet.extend(float_to_bytes_big_endian(target_pitch))
    
    # 包尾：固定标识符 [0x55, 0xAA] - 仅用于识别数据包边界
    packet.extend([0x55, 0xAA])
    
    return packet

def parse_attitude_packet(data_buffer):

    if len(data_buffer) < 22:  # 2+4*4+2 = 22字节
        return None
    
    # 查找包头标识符 [0xBB, 0x66]
    for i in range(len(data_buffer) - 21):
        if data_buffer[i] == 0xBB and data_buffer[i+1] == 0x66:
            # 检查包尾标识符 [0x66, 0xBB]
            if data_buffer[i+20] == 0x66 and data_buffer[i+21] == 0xBB:
                try:
                    # 提取有用数据（跳过包头，只解析中间的float数据）
                    yaw = bytes_to_float_big_endian(data_buffer[i+2:i+6])       
                    pitch = bytes_to_float_big_endian(data_buffer[i+6:i+10])    
                    yaw_rate = bytes_to_float_big_endian(data_buffer[i+10:i+14]) 
                    pitch_rate = bytes_to_float_big_endian(data_buffer[i+14:i+18]) 
                    
                    return {
                        'yaw': yaw,
                        'pitch': pitch,
                        'yaw_rate': yaw_rate,
                        'pitch_rate': pitch_rate
                    }
                except:
                    continue
    return None

def send_target_angles(ser, target_yaw, target_pitch):

    if ser is None or not ser.is_open:
        return
        
    packet = create_target_angle_packet(target_yaw, target_pitch)
    
    # 使用serial库发送数据包
    ser.write(packet)
    
    print(f"yaw: {target_yaw:.2f}°,pitch : {target_pitch:.2f}°")

def read_attitude_data(ser):

    if ser is None or not ser.is_open:
        return None
        
    # 使用serial库读取串口缓冲区中的所有可用数据
    data_buffer = bytearray()
    if ser.in_waiting > 0:  
        data = ser.read(ser.in_waiting) 
        data_buffer.extend(data)
    
    # 解析姿态数据包
    if len(data_buffer) > 0:
        attitude = parse_attitude_packet(data_buffer)
        if attitude:
            return attitude
    
    return None

# 使用余弦定理测量矩形角度，筛选所需四边形
def angle_cos(p0, p1, p2):
    # 转换为正确的形状，即 (2,) 而不是 (1, 2)
    d1, d2 = (p0-p1).squeeze(), (p2-p1).squeeze()
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))

#  如果矩形四个角点距离过近，则判断为同一矩形
def are_rectangles_close(rect1, rect2):
    threshold = 2
    for p1, p2 in zip(rect1, rect2):
        if not (abs(p1[0][0] - p2[0][0]) <= threshold and abs(p1[0][1] - p2[0][1]) <= threshold):
            return False
    return True

#为矩形四个角点排序使用
def sort_vertices(approx):
    # 计算质心
    center = approx.mean(axis=0)
    # 计算每个点相对于质心的角度并排序

    def sort_criteria(point):
        return np.arctan2(point[0][1] - center[0][1], point[0][0] - center[0][0])
    sorted_vertices = sorted(approx, key=sort_criteria)
    return np.array(sorted_vertices)

#找矩形使用，可以检测到嵌套轮廓
def find_rectangles(image):
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 存储检测到的矩形和角点
    rectangles_data = []

    # 处理每一个轮廓
    for cnt in contours:
        # 对轮廓进行近似
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 检查是否为矩形（四个角点）
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            cosines = [angle_cos(approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4]) for i in range(4)]
            if all(cos < 0.1 for cos in cosines):  # 检查角度接近90度
                approx_sorted = sort_vertices(approx)
                too_close = any(are_rectangles_close(approx_sorted, sort_vertices(existing))
                                for existing in rectangles_data)
                if not too_close:
                    # 保存角点
                    rectangles_data.append(approx_sorted)
                    # 在原图上标出矩形
                    cv2.drawContours(img_contour, [approx_sorted], -1, (255, 255, 0), 1)

    return rectangles_data

#  设定红色，绿色HSV色域的阈值
#  实际需要根据环境条件调整
red_hue_low, red_hue_high, red_saturation_1, red_saturation_2, red_value_1, red_value_2 = 150, 15, 70, 70, 70, 70
green_hue_low, green_hue_high, green_saturation, green_value = 50, 90, 90, 90

# 设置红色的阈值范围
# 注意红色在HSV颜色空间中跨0度，可能需要两部分阈值
# 定义红色HSV阈值，同时检测高亮度区域
lower_red1 = np.array([0, red_saturation_1, red_value_1])
upper_red1 = np.array([red_hue_high, 255, 255])
lower_red2 = np.array([red_hue_low, red_saturation_2, red_value_2])
upper_red2 = np.array([180, 255, 255])
# 设定绿色的阈值
lower_green = np.array([green_hue_low, green_saturation, green_value])
upper_green = np.array([green_hue_high, 255, 255])

ret, frame = cap.read()
# 获取图像的维度
height, width = frame.shape[:2]
# 计算中心区域的起始点
start_x = width // 2 - 240
start_y = height // 2 - 240
# 创建一个全黑的遮罩
black_mask = np.zeros((height, width), dtype=np.uint8)
# 在遮罩中心区域填充白色
black_mask[start_y:start_y + 480, start_x:start_x + 470] = 255

# 卡尔曼滤波算法处理激光点坐标，未调参，效果存疑
# 初始化卡尔曼滤波器
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)  # 调整这个参数可以改变模型对系统动态的敏感程度
kalman.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float32)  # 测量噪声
kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
kalman.statePost = np.array([0, 0, 0, 0], np.float32)

def update_kalman_filter(cx, cy):
    # 将当前检测到的位置用作测量更新
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    # 更新卡尔曼滤波器
    kalman.correct(measurement)
    # 预测下一个状态
    predicted = kalman.predict()
    return predicted

def red_laser_detection(image):
    # 创建红色激光掩码
    mask1 = cv2.inRange(image, lower_red1, upper_red1)
    mask2 = cv2.inRange(image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 将遮罩应用到图像
    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=black_mask)

    # 形态学操作增强掩码
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)
    mask_red = cv2.erode(mask_red, kernel, iterations=1)

    mask_canny = cv2.Canny(mask_red, 0, 105)

    cv2.imshow("red_contours", mask_canny)

    contours, hierarchy = cv2.findContours(mask_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化变量以存储最佳轮廓
    best_contour = None
    largest_area = 0

    if hierarchy is not None:
        # 遍历所有轮廓
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # 检查当前轮廓是否有子轮廓
            if hierarchy[0][idx][2] != -1:  # 有子轮廓
                inner_idx = hierarchy[0][idx][2]
                inner_contour = contours[inner_idx]
                inner_area = cv2.contourArea(inner_contour)
                # 使用子轮廓作为最佳轮廓
                if inner_area > largest_area:
                    largest_area = inner_area
                    best_contour = inner_contour
            else:
                # 没有子轮廓，考虑这是一个单独的外部轮廓
                if area > largest_area:
                    largest_area = area
                    best_contour = contour

    # 如果找到了最佳轮廓，计算其质心
    if best_contour is not None:
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img_contour, (cx, cy), 5, (255, 255, 0), -1)
            return cx, cy

def green_laser_detection(image):
    # 根据阈值构建掩码
    green_mask = cv2.inRange(image, lower_green, upper_green)

    # 将遮罩应用到图像
    green_mask = cv2.bitwise_and(green_mask, green_mask, mask=black_mask)
    mask_canny = cv2.Canny(green_mask, 0, 105)
    cv2.imshow('green_mask', mask_canny)
    # 寻找轮廓
    contours, _ = cv2.findContours(mask_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 取最大轮廓
        max_green_laser_contour = max(contours, key=cv2.contourArea)
        (x, y), _ = cv2.minEnclosingCircle(max_green_laser_contour)
        green_laser_point = (int(x), int(y))

        cv2.circle(img_contour, green_laser_point, 5, (0, 0, 255), -1)

        return green_laser_point

# 在黑胶带中心匀速运动的点，即纯闭环扫矩形的目标点
# 函数来计算在矩形边上移动的点的位置
def move_point_on_rectangle(points, t, total_steps):
    # 确定点在哪一条边上
    num_points = len(points)
    # 确定每条边的长度在总步数中的占比
    steps_per_edge = total_steps // num_points
    # 确定当前边
    current_edge = int(t // steps_per_edge) % num_points
    # 当前边的起点和终点
    start_point = points[current_edge]
    end_point = points[(current_edge + 1) % num_points]
    # 计算t在当前边的相对位置
    t_relative = (t % steps_per_edge) / steps_per_edge
    # 计算点的位置
    point_position = start_point + t_relative * (end_point - start_point)
    return point_position


point_level = [320, 240]
velocity = 5
direction = 1

def move_point_level(velocity, direction):
    point_level[0] += velocity * direction
    cv2.circle(img_contour, point_level, 2, (0, 255, 0), thickness=-1)

def callback(x):
    pass

def point_to_line_distance(point, line_start, line_end):
    #计算点到线段的最短距离和最近点
    # 将点和线段转换为numpy数组
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    
    # 计算线段向量和点向量
    ab = b - a
    ap = p - a
    
    # 计算投影参数t
    ab_squared = np.dot(ab, ab)
    if ab_squared == 0:
        # 线段退化为点
        return np.linalg.norm(ap), a
    
    t = np.dot(ap, ab) / ab_squared
    t = max(0, min(1, t))  # 限制在[0,1]范围内
    
    # 计算最近点
    closest_point = a + t * ab
    distance = np.linalg.norm(p - closest_point)
    
    return distance, closest_point

def find_closest_point_on_rectangle(laser_point, rect_vertices):
    # 找到激光点在矩形边界上的最近目标点
    if laser_point is None or len(rect_vertices) != 4:
        return None
    
    min_distance = float('inf')
    closest_point = None
    
    # 遍历矩形的四条边
    for i in range(4):
        start_vertex = rect_vertices[i][0]  # 提取坐标 [x, y]
        end_vertex = rect_vertices[(i + 1) % 4][0]
        
        distance, point_on_edge = point_to_line_distance(
            laser_point, start_vertex, end_vertex
        )
        
        if distance < min_distance:
            min_distance = distance
            closest_point = point_on_edge
    
    return closest_point.astype(int) if closest_point is not None else None

def calculate_rectangle_center(rect_vertices):
    # 计算矩形中心点
    center = np.mean([vertex[0] for vertex in rect_vertices], axis=0)
    return center.astype(int)

def get_target_point_on_rectangle(laser_point, rect_vertices, mode='closest'):
    
    # 根据激光点位置确定矩形上的目标点
    
    # mode选项:
    # - 'closest': 最近点模式 - 激光点投影到最近的矩形边 (正常用这个就行)
    # - 'outward': 向外引导模式 - 如果激光点在矩形内，引导到最近边；如果在外，引导到最近点
    # - 'center_guide': 中心引导模式 - 从矩形中心向激光点方向延伸到边界
    
    
    if laser_point is None or len(rect_vertices) != 4:
        return None
    
    if mode == 'closest':
        return find_closest_point_on_rectangle(laser_point, rect_vertices)
    
    elif mode == 'center_guide':
        # 从矩形中心向激光点方向延伸到边界
        center = calculate_rectangle_center(rect_vertices)
        
        # 计算从中心到激光点的方向向量
        direction = np.array(laser_point) - center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            # 激光点就在中心，返回任意边界点
            return rect_vertices[0][0]
        
        # 归一化方向向量
        direction = direction / direction_norm
        
        # 找到射线与矩形边界的交点
        min_t = float('inf')
        intersection_point = None
        
        for i in range(4):
            start_vertex = np.array(rect_vertices[i][0])
            end_vertex = np.array(rect_vertices[(i + 1) % 4][0])
            
            # 参数方程求交点
            # 射线: center + t * direction
            # 线段: start_vertex + s * (end_vertex - start_vertex)
            
            edge_vector = end_vertex - start_vertex
            
            # 解方程组
            det = direction[0] * edge_vector[1] - direction[1] * edge_vector[0]
            if abs(det) < 1e-6:  # 平行线
                continue
            
            diff = start_vertex - center
            t = (diff[0] * edge_vector[1] - diff[1] * edge_vector[0]) / det
            s = (diff[0] * direction[1] - diff[1] * direction[0]) / det
            
            if t > 0 and 0 <= s <= 1:  # 有效交点
                if t < min_t:
                    min_t = t
                    intersection_point = center + t * direction
        
        return intersection_point.astype(int) if intersection_point is not None else None
    
    else:  # 默认使用closest模式
        return find_closest_point_on_rectangle(laser_point, rect_vertices)

def calculate_tracking_error(laser_point, rect_vertices, tracking_mode='closest'):
    
    # 计算激光点跟踪矩形的error值
    
    # 返回:
    # - error_x: 水平方向error (像素)
    # - error_y: 垂直方向error (像素)
    # - target_point: 目标点坐标 (用于可视化)
   
    
    if laser_point is None or len(rect_vertices) == 0:
        return 0, 0, None
    
    # 获取目标点
    target_point = get_target_point_on_rectangle(laser_point, rect_vertices, tracking_mode)
    
    if target_point is None:
        return 0, 0, None
    
    # 计算error = 目标点 - 当前点
    error_x = target_point[0] - laser_point[0]
    error_y = target_point[1] - laser_point[1]
    
    return error_x, error_y, target_point

# 模式1
def mode1_position_based_tracking(red_laser, rectangles, ser, img_contour):
    
    #基于位置的矩形跟踪模式1
   
    
    if red_laser is None or len(rectangles) < 1:
        return
    
    # 选择矩形（如果有多个，可以选择面积最大的或平均）
    if len(rectangles) >= 2:
        # 使用前两个矩形的平均值
        target_rect = np.average([rectangles[0], rectangles[1]], axis=0).astype(np.int32)
    else:
        # 只有一个矩形
        target_rect = rectangles[0]
    
    # 绘制目标矩形
    cv2.drawContours(img_contour, [target_rect], -1, (255, 255, 0), 2)
    
    # 计算跟踪error
    error_x, error_y, target_point = calculate_tracking_error(
        red_laser, target_rect, tracking_mode='closest'
    )
    
    # 可视化目标点和连接线
    if target_point is not None:
        cv2.circle(img_contour, tuple(target_point), 5, (0, 255, 0), -1)  # 绿色目标点
        cv2.line(img_contour, red_laser, tuple(target_point), (255, 0, 255), 2)  # 紫色连接线
        
        # 显示error信息
        cv2.putText(img_contour, f"Error X:{error_x} Y:{error_y}", 
                   (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img_contour, f"Target:{target_point}", 
                   (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 将像素error转换为角度error并发送
    if target_point is not None:
        # 转换系数：0.03度/像素
        target_yaw = float(error_x * 0.03)    # 水平error
        target_pitch = float(-error_y * 0.03) # 垂直error (注意负号)
        
        # 发送error到下位机
        send_target_angles(ser, target_yaw, target_pitch)
        
        print(f"Position-based tracking - Error X:{error_x}, Y:{error_y}, Yaw:{target_yaw:.3f}°, Pitch:{target_pitch:.3f}°")

cv2.namedWindow('Color Adjustments', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Color Adjustments', 600, 400)

# 创建滑条，每个参数一个
# 红色的色调、饱和度和亮度值
cv2.createTrackbar('Red Hue Low', 'Color Adjustments', 150, 180, callback)
cv2.createTrackbar('Red Hue High', 'Color Adjustments', 15, 180, callback)
cv2.createTrackbar('Red Saturation 1', 'Color Adjustments', 70, 255, callback)
cv2.createTrackbar('Red Saturation 2', 'Color Adjustments', 70, 255, callback)
cv2.createTrackbar('Red Value 1', 'Color Adjustments', 70, 255, callback)
cv2.createTrackbar('Red Value 2', 'Color Adjustments', 70, 255, callback)

# 绿色的色调、饱和度和亮度值
cv2.createTrackbar('Green Hue Low', 'Color Adjustments', 50, 180, callback)
cv2.createTrackbar('Green Hue High', 'Color Adjustments', 90, 180, callback)
cv2.createTrackbar('Green Saturation', 'Color Adjustments', 90, 255, callback)
cv2.createTrackbar('Green Value', 'Color Adjustments', 90, 255, callback)

# 创建颜色块以显示当前设置的颜色
color_block = np.zeros((300, 300, 3), np.uint8)

#  计时用
# t = 0
# total_steps = 300

track_point = (320, 240)

try:
    ser = serial.Serial('/dev/ttyACM0', 921600, timeout=0.1)
    print("串口打开成功")
except:
    ser = None
    print("串口打开失败")

# 用于存储接收到的姿态数据
current_attitude = None

while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv2.imread("/home/orangepi/Desktop/photo.png")
    img = frame
    img_contour = img

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img_contour, f"fps:{fps}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    # 读取下位机发送的姿态数据
    attitude_data = read_attitude_data(ser)  # 修改：传入ser对象
    if attitude_data:
        current_attitude = attitude_data
        print(f"Received attitude - Yaw: {attitude_data['yaw']:.2f}, Pitch: {attitude_data['pitch']:.2f}")

    # 显示当前姿态数据
    if current_attitude:
        cv2.putText(img_contour, f"Attitude Y:{current_attitude['yaw']:.1f} P:{current_attitude['pitch']:.1f}", 
                    (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    # 预处理图像，包括转为灰度图，高斯模糊，用canny算子边缘检测，以及检测激光用的HSV色域图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 50, 150)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #找矩形，红激光，绿激光
    rectangles = find_rectangles(img_canny)
    red_laser = red_laser_detection(img_hsv)

    # 卡尔曼滤波预测激光点位置，未调参，效果存疑
    if red_laser is not None:
        red_laser_cx, red_laser_cy = red_laser
        predicted_red_laser = update_kalman_filter(red_laser_cx, red_laser_cy)
        predicted_red_laser_position = (int(predicted_red_laser[0]), int(predicted_red_laser[1]))
        cv2.circle(img_contour, predicted_red_laser_position, 3, (255, 0, 0), -1)
    green_laser = green_laser_detection(img_hsv)

    #显示分辨率
    cv2.putText(img_contour, f"resolution:{width}*{height}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    if rectangles:
        for rect in rectangles:
            print("rectangle:", rect)
    if red_laser:
        print("red:", red_laser)
        cv2.putText(img_contour, f"red:{red_laser}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    if green_laser:
        print("green", green_laser)
        cv2.putText(img_contour, f"green:{green_laser}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    if red_laser is not None and green_laser is not None and mode == 0:
        
        error_x = predicted_red_laser[0] - green_laser[0]
        error_y = predicted_red_laser[1] - green_laser[1]
        
        # 将像素偏差转换为角度偏差
        target_yaw = float(error_x *0.03 ) 
        target_pitch = float(-error_y *0.03 )
        
        cv2.putText(img_contour, f"target YAW:{target_yaw:.2f} PITCH:{target_pitch:.2f}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        
        # 发送目标角度
        send_target_angles(ser, target_yaw, target_pitch) 
        

    if (mode == 1) and len(rectangles) >= 2:
        mode1_position_based_tracking(red_laser, rectangles, ser, img_contour)

    if mode == 2 and red_laser is not None:
        move_point_level(velocity, direction)
        if point_level[0] <= 150 or point_level[0] >= 500:
            direction *= -1
        
        error_x = red_laser[0] - point_level[0]
        error_y = red_laser[1] - point_level[1]
        
        # 将像素偏差转换为角度偏差
        target_yaw = float(error_x * 0.03)
        target_pitch = float(-error_y * 0.03)
        
        # 发送目标角度
        send_target_angles(ser, target_yaw, target_pitch)  # 修改：传入ser对象

    if mode == 3 and red_laser is not None:
        cv2.circle(img_contour, track_point, 3, (0, 255, 0), thickness=-1)
        
        error_x = red_laser[0] - track_point[0]
        error_y = red_laser[1] - track_point[1]
        
        # 将像素偏差转换为角度偏差
        target_yaw = float(error_x * 0.03)
        target_pitch = float(-error_y * 0.03)
        
        # 发送目标角度
        send_target_angles(ser, target_yaw, target_pitch)  # 修改：传入ser对象
    
    # 滑条调参，获取滑条的值
    red_hue_low = cv2.getTrackbarPos('Red Hue Low', 'Color Adjustments')
    red_hue_high = cv2.getTrackbarPos('Red Hue High', 'Color Adjustments')
    red_saturation_1 = cv2.getTrackbarPos('Red Saturation 1', 'Color Adjustments')
    red_saturation_2 = cv2.getTrackbarPos('Red Saturation 2', 'Color Adjustments')
    red_value_1 = cv2.getTrackbarPos('Red Value 1', 'Color Adjustments')
    red_value_2 = cv2.getTrackbarPos('Red Value 2', 'Color Adjustments')
    green_hue_low = cv2.getTrackbarPos('Green Hue Low', 'Color Adjustments')
    green_hue_high = cv2.getTrackbarPos('Green Hue High', 'Color Adjustments')
    green_saturation = cv2.getTrackbarPos('Green Saturation', 'Color Adjustments')
    green_value = cv2.getTrackbarPos('Green Value', 'Color Adjustments')
    lower_red1 = np.array([0, red_saturation_1, red_value_1])
    upper_red1 = np.array([red_hue_high, 255, 255])
    lower_red2 = np.array([red_hue_low, red_saturation_2, red_value_2])
    upper_red2 = np.array([180, 255, 255])
    # 设定绿色的阈值
    lower_green = np.array([green_hue_low, green_saturation, green_value])
    upper_green = np.array([green_hue_high, 255, 255])
    # 根据HSV值更新颜色块显示
    color_block[:] = [((red_hue_low + green_hue_low) // 2, (red_saturation_1 + green_saturation) // 2,
                       (red_value_1 + green_value) // 2)]
    color_block = cv2.cvtColor(color_block, cv2.COLOR_HSV2BGR)

    # 显示颜色块
    cv2.imshow('Color Adjustments', color_block)

    cv2.imshow("frame", img_contour)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('z'):
        track_point = (200, 240)
    if key == ord('x'):
        track_point = (480, 240)

    # t += 1
    # if t == total_steps:
    #     t = 0
