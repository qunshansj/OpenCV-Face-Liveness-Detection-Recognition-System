python

class Detector:
    def __init__(self, weights, device='cpu', imgsz=640, conf_thres=0.7, iou_thres=0.45):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = None
        self.names = None
        self.colors = None

    def load_model(self):
        set_logging()
        self.device = select_device(self.device)
        half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, map_location=self.device)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())
        if half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def predict(self, im0s):
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None
        img = self.letterbox(im0s, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        ret = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]}'
                    prob = round(float(conf) * 100, 2)
                    ret_i = [label, prob, xyxy]
                    ret.append(ret_i)
        return ret
