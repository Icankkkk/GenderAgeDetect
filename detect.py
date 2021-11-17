import cv2
import argparse

def highlightFace(net, frame, conf_threshold = 0.7):
    frameOpenCvDnn = frame.copy()
    frameHeight = frameOpenCvDnn.shape[0]
    frameWidth = frameOpenCvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpenCvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    areaWajah = []
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 =int(detections[0, 0, i, 3] * frameWidth)
            y1 =int(detections[0, 0, i, 4] * frameHeight)
            x2 =int(detections[0, 0, i, 5] * frameWidth)
            y2 =int(detections[0, 0, i, 6] * frameHeight)
            areaWajah.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpenCvDnn, (x1, y1), (x2, y2), (255, 102, 204), int(round(frameHeight / 150)), 8)
    return frameOpenCvDnn, areaWajah

argumen = argparse.ArgumentParser()
argumen.add_argument('--foto')

# memanggil method
args = argumen.parse_args()

protoWajah = "opencv_face_detector.pbtxt"
modelWajah = "opencv_face_detector_uint8.pb"
umurProto = "age_deploy.prototxt"
modelUmur = "age_net.caffemodel"
jenisKelaminProto = "gender_deploy.prototxt" 
jenisKelaminModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# daftar umur
listUmur = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
# daftar jenis kelamin
listJk = ['Cowo','Cewe'] 

wajahNet = cv2.dnn.readNet(modelWajah, protoWajah)
umurNet = cv2.dnn.readNet(modelUmur, umurProto)
jenisKelaminNet = cv2.dnn.readNet(jenisKelaminModel, jenisKelaminProto)

video = cv2.VideoCapture(args.foto if args.foto else 0)
padding = 20
while cv2.waitKey(1) < 0 :
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    hasilGambar, areaWajah = highlightFace(wajahNet, frame)
    if not areaWajah:
        print("tidak ada wajah yang terdeteksi")

    for wajahBox in areaWajah:
        wajah = frame[max(0, wajahBox[1]-padding):
                   min(wajahBox[3]+padding,frame.shape[0]-1),max(0,wajahBox[0]-padding)
                   :min(wajahBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(wajah, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        jenisKelaminNet.setInput(blob)
        jkPrediksi = jenisKelaminNet.forward()
        jenisKelamin = listJk[jkPrediksi[0].argmax()]
        print(f'Jenis kelamin: {jenisKelamin}')

        umurNet.setInput(blob)
        umurPrediksi = umurNet.forward()
        umur = listUmur[umurPrediksi[0].argmax()]
        print(f'Umur: {umur[1:-1]} tahun')

        cv2.putText(hasilGambar, f'{jenisKelamin}, {umur}', (wajahBox[0], wajahBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Mendeteksi umur dan jenis kelamin", hasilGambar)
