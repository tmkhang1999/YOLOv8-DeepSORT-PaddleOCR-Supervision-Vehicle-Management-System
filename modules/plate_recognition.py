def recognize_plate(license_plate_detector, ocr_model, frame, xyxy, plate_conf=0.6):
    x1, y1, x2, y2 = [int(i) for i in xyxy]
    car_frame = frame[int(y1):int(y2), int(x1):int(x2)]
    plates = license_plate_detector(car_frame, conf=plate_conf)

    if plates[0]:
        x1, y1, x2, y2 = plates[0].boxes.xyxy.cpu().numpy()[0]
        plate_frame = car_frame[int(y1):int(y2), int(x1):int(x2)]

        result = ocr_model.ocr(plate_frame, cls=True)
        if result[0]:
            info = result[0][0][-1]
            return info[0], info[1]

    return None, None
