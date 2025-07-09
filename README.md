上位机部分
香橙派3b opencv编写
下位机 
达妙板 搭载STM32H723 
使用电机而非舵机，纯闭环方案
云台imu自稳，接受上位机error值闭环

视觉思路
灰度图高斯模糊处理，Canny找轮廓，轮廓近似后取角点对角度判断
放置LAB色彩遮罩{220,255}，Canny遍历光斑外接圆，记录center
LAB红绿分别遮罩，Canny取最小外接圆，若为空集就取center
serial通信下位机，yaw error，pitch error
