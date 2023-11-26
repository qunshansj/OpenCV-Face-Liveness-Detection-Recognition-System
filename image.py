python


class FaceDetector:
    def __init__(self):
        self.test_mode = "onet"
        self.thresh = [0.9, 0.6, 0.7]
        self.min_face_size = 24
        self.stride = 2
        self.slide_window = False
        self.shuffle = False
        self.detectors = [None, None, None]
        self.prefix = ['./data/MTCNN_model/PNet_landmark/PNet', './data/MTCNN_model/RNet_landmark/RNet', './data/MTCNN_model/ONet_landmark/ONet']
        self.epoch = [18, 14, 16]
        self.model_path = ['%s-%s' % (x, y) for x, y in zip(self.prefix, self.epoch)]
        self.PNet = FcnDetector(P_Net, self.model_path[0])
        self.detectors[0] = self.PNet
        self.RNet = Detector(R_Net, 24, 1, self.model_path[1])
        self.detectors[1] = self.RNet
        self.ONet = Detector(O_Net, 48, 1, self.model_path[2])
        self.detectors[2] = self.ONet

    def detect_faces(self, imagepath):
        mtcnn_detector = MtcnnDetector(detectors=self.detectors, min_face_size=self.min_face_size,
                                       stride=self.stride, threshold=self.thresh, slide_window=self.slide_window)
        corpbbox = None
        frame = cv2.imread(imagepath)
        boxes_c,landmarks = mtcnn_detector.detect(frame)
        print(landmarks.shape)
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (0,0,255), 1)
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        for i in range(landmarks.shape[0]):
            for j in range(int(len(landmarks[i])/2)):
                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 4,(0,0,255),-1)
        # time end
        cv2.imshow("output", frame)
        cv2.waitKey(0)

if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.detect_faces("./face_dataset/test3.png")
