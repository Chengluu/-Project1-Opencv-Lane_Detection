import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from moviepy.editor import VideoFileClip

"参数设置"
nx = 9
ny = 6
file_paths = glob.glob("./Camera_calibration/calibration*.jpg")

# 绘制对比图
def plot_contrast_image(origin_img,converted_img,origin_img_title='origin_img',converted_img_title='converted_img',converted_img_gray=False):
	fig,(ax1,ax2) = plt.subplots(1,2,figsize=(30,50))
	ax1.set_title = origin_img_title
	ax1.imshow(origin_img)
	ax2.set_title = converted_img_title
	if converted_img_gray == True:
		ax2.imshow(converted_img,cmap='gray')
	else:
		ax2.imshow(converted_img)
	plt.show()
# 相机矫正：外参，内参，畸变系数
def cal_calibrate_params(file_paths):
	# 存储角点数据的坐标
	object_points = []
	image_points = []
	objp = np.zeros((nx * ny, 3), np.float32)
	objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
	# 2.2 检测每幅图像⻆点坐标
	for file_path in file_paths:
		img = cv2.imread(file_path)
		# 将图像转换为灰度图
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# ⾃动检测棋盘格内4个棋盘格的⻆点（2⽩2⿊的交点）
		rect, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		imgcopy = img.copy()
		cv2.drawChessboardCorners(imgcopy,(nx,ny),corners,rect)
		plot_contrast_image(img,imgcopy)
		# 若检测到⻆点，则将其存储到object_points和image_points
		if rect == True:
			object_points.append(objp)
			image_points.append(corners)
	# 2.3 获取相机参数
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,image_points,gray.shape[::-1],None,None)
	return ret, mtx, dist, rvecs, tvecs

def img_undistort(img,mtx,dist):
	dis = cv2.undistort(img,mtx,dist,None,mtx)
	return dis

def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
	img = np.copy(img)
	#1.将图像转换为HLS⾊彩空间，并分离各个通道
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	h_channel = hls[:, :, 0]
	l_channel = hls[:, :, 1]
	s_channel = hls[:, :, 2]
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	# 将导数转换为8bit整数
	scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	# 3.对s通道进⾏阈值处理
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	# 4. 将边缘检测的结果和颜⾊空间阈值的结果合并，并结合l通道的取值，确定⻋道提取的⼆值图结
	color_binary = np.zeros_like(sxbinary)
	color_binary[((sxbinary == 1) | (s_binary == 1)) & (l_channel > 100)] = 1
	return color_binary

def cal_perspective_params(img, points):
	offset_x = 330
	offset_y = 0
	img_size = (img.shape[1], img.shape[0])
	src = np.float32(points)
	# 俯视图中四点的位置
	dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y], [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
	# 从原始图像转换为俯视图的透视变换的参数矩阵
	M = cv2.getPerspectiveTransform(src, dst)
	# 从俯视图转换为原始图像的透视变换参数矩阵
	M_inverse = cv2.getPerspectiveTransform(dst, src)
	return M, M_inverse

def img_perspect_transform(img, M):
	img_size = (img.shape[1], img.shape[0])
	return cv2.warpPerspective(img, M, img_size)


def cal_line_param(binary_warped):
	# 1.确定左右车道线的位置
	# 统计直方图
	histogram = np.sum(binary_warped[:, :], axis=0)
	# 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
	# 将统计结果一分为二，划分为左右两个部分，分别定位峰值位置，即为两条车道的搜索位置
	midpoint = np.int(histogram.shape[0] / 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# 2.滑动窗口检测车道线
	# 设置滑动窗口的数量，计算每一个窗口的高度
	nwindows = 9
	window_height = np.int(binary_warped.shape[0] / nwindows)
	# 获取图像中不为0的点
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# 车道检测的当前位置
	leftx_current = leftx_base
	rightx_current = rightx_base
	# 设置x的检测范围，滑动窗口的宽度的一半，手动指定
	margin = 100
	# 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
	minpix = 50
	# 用来记录搜索窗口中非零点在nonzeroy和nonzerox中的索引
	left_lane_inds = []
	right_lane_inds = []

	# 遍历该副图像中的每一个窗口
	for window in range(nwindows):
		# 设置窗口的y的检测范围，因为图像是（行列）,shape[0]表示y方向的结果，上面是0
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		# 左车道x的范围
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		# 右车道x的范围
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# 确定非零点的位置x,y是否在搜索窗口中，将在搜索窗口内的x,y的索引存入left_lane_inds和right_lane_inds中
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# 将检测出的左右车道点转换为array
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# 获取检测出的左右车道点在图像中的位置
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	return left_fit, right_fit

def fill_lane_poly(img, left_fit, right_fit):
	y_max = img.shape[0]
	out_img = np.dstack((img,img,img))*255
	left_points = [[left_fit[0]*y**2+left_fit[1]*y+left_fit[2],y] for y in range(y_max)]
	right_points = [[right_fit[0]*y**2+right_fit[1]*y+right_fit[2],y] for y in range(y_max-1,-1,-1)]
	line_points = np.vstack((left_points, right_points))
	# 根据左右车道线的像素位置绘制多边形
	cv2.fillPoly(out_img, np.int_([line_points]), (0, 255, 0))
	return out_img

# 计算车道线曲率的方法
def cal_radius(img, left_fit, right_fit):
	# 比例
	ym_per_pix = 30 / 720
	xm_per_pix = 3.7 / 700
	# 得到车道线上的每个点
	left_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)  # 个数img.shape[0]-1
	left_x_axis = left_fit[0] * left_y_axis ** 2 + left_fit[1] * left_y_axis + left_fit[0]
	right_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
	right_x_axis = right_fit[0] * right_y_axis ** 2 + right_fit[1] * right_y_axis + right_fit[2]

	# 把曲线中的点映射真实世界，再计算曲率
	# polyfit(x,y,n)。用多项式求过已知点的表达式，其中x为源数据点对应的横坐标，可为行 向 量、矩阵，
	# y为源数据点对应的纵坐标，可为行向量、矩阵，
	# n为你要拟合的阶数，一阶直线拟合，二阶抛物线拟合，并非阶次越高越好，看拟合情况而定
	left_fit_cr = np.polyfit(left_y_axis * ym_per_pix, left_x_axis * xm_per_pix, 2)
	right_fit_cr = np.polyfit(right_y_axis * ym_per_pix, right_x_axis * xm_per_pix, 2)
	# 计算曲率
	left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_axis * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
	right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_axis * ym_per_pix * right_fit_cr[1]) ** 2) ** 1.5) / np.absolute((2 * right_fit_cr[0]))

	# 将曲率半径渲染在图像上 写什么
	cv2.putText(img, 'Radius of Curvature = {}(m)'.format(np.mean(left_curverad)), (20, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	return img


# 计算车道线中心的位置
def cal_line_center(img):
	# 去畸变
	undistort_img = img_undistort(img, mtx, dist)
	# 提取车道线
	rigin_pipeline_img = pipeline(undistort_img)
	# 透视变换
	trasform_img = img_perspect_transform(rigin_pipeline_img, M)
	# 精确定位
	left_fit, right_fit = cal_line_param(trasform_img)
	# 当前图像的shape[0]
	y_max = img.shape[0]
	# 左车道线
	left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
	# 右车道线
	right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
	# 返回车道中心点
	return (left_x + right_x) / 2


# 计算中心点
def cal_center_departure(img, left_fit, right_fit):
	y_max = img.shape[0]
	left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
	right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
	xm_per_pix = 3.7 / 700
	lane_center = cal_line_center(img)
	center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
	# 渲染
	if center_depart > 0:
		cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart), (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	elif center_depart < 0:
		cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart), (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	else:
		cv2.putText(img, 'Vehicle is in the center', (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	return img


# 计算车辆偏离中心点的距离
def cal_center_departure(img, left_fit, right_fit):
	# 计算中心点
	y_max = img.shape[0]
	# 左车道线
	left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
	# 右车道线
	right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
	# x方向上每个像素点代表的距离大小
	xm_per_pix = 3.7 / 700
	# 计算偏移距离 像素距离 × xm_per_pix = 实际距离
	lane_center = cal_line_center(img)
	center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
	# 渲染
	if center_depart > 0:
		cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart), (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	elif center_depart < 0:
		cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart), (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	else:
		cv2.putText(img, 'Vehicle is in the center', (20, 100), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
	return img


def process_image(img):
	# 1.图像去畸变
	undistort_img = img_undistort(img, mtx, dist)
	# 2.⻋道线检测
	rigin_pipline_img = pipeline(undistort_img)
	# 3.透视变换
	transform_img = img_perspect_transform(rigin_pipline_img, M)
	# 4.精确定位⻋道线，并拟合
	left_fit, right_fit = cal_line_param(transform_img)
	# 5.绘制⻋道区域
	result = fill_lane_poly(transform_img, left_fit, right_fit)

	# 转换回原来的视角
	transform_img_inv = img_perspect_transform(result, M_inverse)

	# 曲率和偏离距离
	transform_img_inv = cal_radius(transform_img_inv, left_fit, right_fit)
	# 偏离距离
	transform_img_inv = cal_center_departure(transform_img_inv, left_fit, right_fit)
	# 附加到原图上
	transform_img_inv = cv2.addWeighted(undistort_img, 1, transform_img_inv, 0.5, 0)
	# 返回处理好的图像
	return transform_img_inv

if __name__ == '__main__':
	ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(file_paths)
	# if np.all(mtx!=None):
	# 	img = cv2.imread(".test/test1.jpg")
	# 	undistort_img = img_undistort(img,mtx,dist)
	# 	plot_contrast_image(img,undistort_img)
	# 	print("done")
	# else:
	# 	print("failed")

	# #测试车道线提取
	# result = pipeline(img)
	# plot_contrast_image(img,result,converted_img_gray=True)
	img = cv2.imread('./test/test1.jpg')
	points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
	img = cv2.line(img, (601, 448), (683, 448), (0, 0, 255), 3)
	img = cv2.line(img, (683, 448), (1097, 717), (0, 0, 255), 3)
	img = cv2.line(img, (1097, 717), (230, 717), (0, 0, 255), 3)
	img = cv2.line(img, (230, 717), (601, 448), (0, 0, 255), 3)
	M, M_inverse = cal_perspective_params(img, points)
	lane_center = cal_line_center(img)
	# transform_img = img_perspect_transform(img, M)
	# plt.figure(figsize=(20, 8))
	# plt.subplot(1, 2, 1)
	# plt.title('Original image')
	# plt.imshow(img[:, :, ::-1])
	# plt.subplot(1, 2, 2)
	# plt.title('Top view')
	# plt.imshow(transform_img[:, :, ::-1])
	# plt.show()
	#
	# undistort_img = img_undistort(img, mtx, dist)
	# # 提取车道线
	# pipeline_img = pipeline(undistort_img)
	# # 透视变换
	# trasform_img = img_perspect_transform(pipeline_img, M)
	# # 计算车道线的拟合结果
	# left_fit, right_fit = cal_line_param(trasform_img)
	# # 进行填充
	# result = fill_lane_poly(trasform_img, left_fit, right_fit)
	# plt.figure()
	# # 反转CV2中BGR 转化为matplotlib的RGB
	# plt.imshow(result[:, :, ::-1])
	# plt.title("vertical view:FULL")
	# plt.show()
	#
	# # 透视变换的逆变换
	# trasform_img_inv = img_perspect_transform(result, M_inverse)
	# plt.figure()
	# # 反转CV2中BGR 转化为matplotlib的RGB
	# plt.imshow(trasform_img_inv[:, :, ::-1])
	# plt.title("Original drawing:FULL")
	# plt.show()
	#
	# # 与原图进行叠加
	# res = cv2.addWeighted(img, 1, trasform_img_inv, 0.5, 0)
	# plt.figure()
	# # 反转CV2中BGR 转化为matplotlib的RGB
	# plt.imshow(res[:, :, ::-1])
	# plt.title("safe work")
	# plt.show()

	clip1 = VideoFileClip("./test/test1.mp4")
	white_clip = clip1.fl_image(process_image)
	white_clip.write_videofile("./test/output.mp4", audio=False)