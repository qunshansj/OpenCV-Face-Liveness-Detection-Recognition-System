python


class FatigueDetection:
    def __init__(self):
        # 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
        print("[INFO] loading facial landmark predictor...")
        # 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')
        # 分别获取左右眼面部标志的索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def detect_fatigue(self, frame):
        #frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用detector(gray, 0) 进行脸部位置检测
        rects = self.detector(gray, 0)
        eyear = 0.0
        mouthar = 0.0
        # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
        for rect in rects:
            shape = self.predictor(gray, rect)

            # 将脸部特征信息转换为数组array的格式
            shape = face_utils.shape_to_np(shape)

            # 提取左眼和右眼坐标
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            # 嘴巴坐标
            mouth = shape[self.mStart:self.mEnd]

            # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            eyear = (leftEAR + rightEAR) / 2.0
            # 打哈欠
            mouthar = self.mouth_aspect_ratio(mouth)

            # 标注识别结果
            # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # 画出眼睛、嘴巴竖直线
            cv2.line(frame,tuple(shape[38]),tuple(shape[40]),(0, 255, 0), 1)
            cv2.line(frame,tuple(shape[43]),tuple(shape[47]),(0, 255, 0), 1)
            cv2.line(frame,tuple(shape[51]),tuple(shape[57]),(0, 255, 0), 1)
            cv2.line(frame,tuple(shape[48]),tuple(shape[54]),(0, 255, 0), 1)

        # 返回信息
        # frame已经标注出眼睛和嘴巴的框线
        # eyeae为眼睛的长宽比
        # mouthar为嘴巴的长宽比
        return(frame, eyear, mouthar)
