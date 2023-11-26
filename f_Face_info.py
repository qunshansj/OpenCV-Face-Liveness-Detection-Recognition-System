python

class FaceRecognition:
    def __init__(self):
        self.rec_face = f_main.rec()

    def get_face_info(self, im):
        # face detection
        boxes_face = face_recognition.face_locations(im)
        out = []
        if len(boxes_face)!=0:
            for box_face in boxes_face:
                # segmento rostro
                box_face_fc = box_face
                x0,y1,x1,y0 = box_face
                box_face = np.array([y0,x0,y1,x1])
                face_features = {
                    "name":[],
                    "bbx_frontal_face":box_face
                }
                face_image = im[x0:x1,y0:y1]
                # -------------------------------------- face_recognition ---------------------------------------
                face_features["name"] = self.rec_face.recognize_face2(im,[box_face_fc])[0]

                # -------------------------------------- out ---------------------------------------
                out.append(face_features)
        else:
            face_features = {
                "name":[],
                "bbx_frontal_face":[]
            }
            out.append(face_features)
        return out

    def bounding_box(self, out,img):
        for data_face in out:
            box = data_face["bbx_frontal_face"]
            if len(box) == 0:
                continue
            else:
                x0,y0,x1,y1 = box
                img = cv2.rectangle(img,
                                (x0,y0),
                                (x1,y1),
                                (0,0,255),2);
                thickness = 2
                fontSize = 0.6
                step = 13
                try:
                    cv2.putText(img, "name: " +data_face["name"], (x0, y0-step-7), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
                except:
                    pass
        return img
