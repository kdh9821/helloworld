#!/usr/bin/env python3
# v.1.2 211117
# 기본적 손 인식 기술 구현 완료

import cv2
import mediapipe as mp
import numpy as np

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok'
}   #MediaPipe 제공 제스쳐
hand_gesture = {
    0:'fist', 1:'one', 2:'gun', 3:'three', 4:'four', 5:'five',
    6:'promise', 7:'spiderman', 8:'niconiconi', 9:'two', 10:'ok',
    11:'claws', 12:'good', 13:'illionaire', 14:'dog'
}   #게임에 사용할 제스처 세트 -> 아직 적용안됨

#MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils #웹캠에서 손가락 뼈마디 부분을 그리는 것
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  #모드 세팅

#Gesture recognition model
file = np.genfromtxt('gesture_train_original.csv', delimiter=',')    #csv 파일 받아와서 필요한 정보 뽑기
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()  #KNN(K-Nearest Neighbors) 알고리즘을 통해 손모양 학습?
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cam = cv2.VideoCapture(0) #캠켜기


while cam.isOpened():   #카메라가 열려있으면..
    success, image = cam.read() #한 프레임 씩 읽어옴
    if not success: #success 못하면 다음 프레임으로..?
        continue
    #success하면 go
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #이미지 전처리(색상 형식 변경 & 이미지 한번 뒤집기)
    results = hands.process(image)  #전처리 및 모델 추론을 함께 실행..
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #출력을 위해 다시 색상 형식 바꿔주기

    if results.multi_hand_landmarks:    #위 전처리를 통해 손이 인식 되면 참이됨
        for hand_landmarks in results.multi_hand_landmarks: #손 여러개 대비?? 예외처리 방지? with 써야되나?
            joint = np.zeros((21, 3))   #joint -> 빨간 점. 포인트 21개, xyz 3개. 생성
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]   #값 입력
                
            #joint 인덱스끼리 빼줘서 뼈대의 벡터 구하기(Fig 3의 형태)
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            #벡터의 길이로.. Normalize v?
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            
            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                
            angle = np.degrees(angle) # Convert radian to degree
            
            # Inference gesture / 데이터 바꿔주고 정리..
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            

            cv2.putText(image, text = gesture[idx].upper(), org=(20, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = 255, thickness = 3)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #마디마디에 그려주는
    
    cv2.imshow('Hand Cam', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
