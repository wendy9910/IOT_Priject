# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import pandas as pd
import datetime

# construct the argument parser and parse the 
#構造參數解析器並解析參數
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
#從磁盤加載序列化的面部檢測器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
#從磁盤加載序列化的面部嵌入模型
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
# 加載實際的人臉識別模型以及標籤編碼器
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
# 初始化視頻流，然後讓攝像頭傳感器預熱
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(5.0)

# start the FPS throughput estimator
fps = FPS().start()


tm = datetime.datetime.today()
date_str = tm.strftime("%Y/%m/%d")
tm_str = tm.strftime("%H:%M:%S")

data = pd.read_excel(os.path.join('Staffprofile.xlsx'),engine='openpyxl')
Signdata = pd.read_excel(os.path.join('SignIn.xlsx'),engine='openpyxl')
			
# loop over frames from the video file stream
# 循環播放視頻文件流中的幀
while True:
	# grab the frame from the threaded video stream
    # 從線程視頻流中抓取幀
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
    #將框架調整為600像素的寬度
    
	# maintaining the aspect ratio), and then grab the image
    # 保持寬高比，然後抓取圖像
    
	# dimensions
    
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image 從圖像構造斑點
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
    # 應用OpenCV的基於深度學習的面部檢測器進行本地化
    
	# faces in the input image
    # 輸入圖像中的面孔
    
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
    # 循環檢測
    
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
        # 過濾掉弱檢測
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
            # 計算邊界框的（x，y）坐標
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
            # 提取面部ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
            # 確保臉的寬度和高度足夠大
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
            # 為面部ROI構建斑點，然後通過斑點
			# through our face embedding model to obtain the 128-d
            # 通過我們的人臉嵌入模型獲得128-d
			# quantification of the face 臉部量化
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face 進行分類以識別人臉
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]


			
			if proba*100 >60:

			#員工認證通道

				
				data.loc[(data['員工姓名'] == name) & (data['部門代號'] == 'A02'), '權限'] = "Pass"
				#data.loc[(data['權限'] == 'Pass') & (data['員工姓名'] == name),
					 #'打卡:%s' % (date_str)] = tm_str
				Signdata.loc[(Signdata['員工姓名'] == name) & (data['權限'] == 'Pass' ) , '打卡:%s' %(date_str) ] = tm_str


				pd.set_option('display.unicode.ambiguous_as_wide', True)
				pd.set_option('display.unicode.east_asian_width', True)
				pd.set_option('display.width', 200) # 设置打印宽度(**重要**)


				df = pd.DataFrame(data)
				data.to_excel('Staffprofile.xlsx',index=0)
				Signdata.to_excel('SignIn.xlsx',index=0)

				print(data)
				print(Signdata)
				print(name)


			else:
				data.loc[data['權限']== "Error"]			

			# draw the bounding box of the face along with the
			# associated probability
            		# 繪製臉部的邊界框以及相關的概率
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter 更新FPS計數器
	fps.update()

	# show the output frame 顯示輸出框
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
