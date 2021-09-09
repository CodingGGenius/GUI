import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
import tkinter as tk
import cv2 
import datetime
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext
import numpy as np
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model

right_arm_model = load_model('./models/right_arm_model.h5')
left_arm_model = load_model('./models/left_arm_model.h5')
right_leg_model = load_model('./models/right_leg_model.h5')
left_leg_model = load_model('./models/left_leg_model.h5')

device = "gpu" # please change it to "gpu" if the model needs to be run on cuda.

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# Find valid connections between the different joints of a all persons present
def getValidPairs(output, frameWidth, frameHeight, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
#             print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
#     print(valid_pairs)
    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def pred_multiple_onvideo(video_path, filename):
    if video_path == 'cam':
        cap = cv2.VideoCapture(int(filename))
    else:
        cap = cv2.VideoCapture(video_path)
        print(video_path)
    data = []
    position = []
    last_position = []
    final_position_index = -1
    detected_humans = [] #[human][index, part_seq, part_nan_count, part_action, count]
    
    ret, image1 = cap.read()
    while cap.isOpened():
        t0 = cv2.getTickCount()
        if ret == False:
            break
            
        frameWidth = image1.shape[1]
        frameHeight = image1.shape[0]

        # Fix the input Height and get the width according to the Aspect Ratio
        inHeight = 368 #인식에 문제가 있다면 이 값을 크게 조절하세요 (max = videofile height)
        inWidth = int((inHeight/frameHeight)*frameWidth)
        inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)
        output = net.forward()

        i = 0
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        detected_keypoints = []
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            keypoints = getKeypoints(probMap, threshold)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(output, frameWidth, frameHeight, detected_keypoints)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)
        
        
        #사람 구분
        detected_Obj = []
        last_position += position
        position = []
        keylist = np.empty((len(personwiseKeypoints), 13, 2))

        for i in range(1, 14):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                A = np.int32(keypoints_list[personwiseKeypoints[n][i].astype(int)])
                keylist[n][i - 1] = A[:-1] #keylist는 목부터 발까지 좌표 값만 저장됨

        for n in range(len(personwiseKeypoints)):
            short_distance = 10000
            if int(keylist[n][0][0]) == 0:
                continue
            else:
                #여기서 벡터값 계산
                joint = np.empty((13, 2))
                for j in range(13):
                    try:
                        joint[j] = [keylist[n][j][0], keylist[n][j][1]]
                    except:
                        pass

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11], :2] # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], :2] # Child joint
                v = v2 - v1 # [12, 2]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0, 1, 3, 4, 6, 7, 9, 10],:], 
                    v[[1, 2, 4, 5, 7, 8, 10, 11],:]))

                angle = np.degrees(angle) # Convert radian to degree

                angle_label = np.array([angle], dtype=np.float32)

                count = -1
                last_p_index = -1
                short_p_index = -1
                for last_p in range(len(last_position)):
                    temp_x = keylist[n][0][0] - last_position[last_p][0]
                    temp_y = keylist[n][0][1] - last_position[last_p][1]
                    temp_distance = temp_x * temp_x + temp_y * temp_y
                    if temp_distance < short_distance:
                        short_distance = temp_distance
                        last_p_index = last_p
                        position_index = last_position[last_p][2]
                        
                if last_p_index == -1: #전 프레임에 같은 사람이 없을때
                    final_position_index+=1
                    position.append([keylist[n][0][0], keylist[n][0][1], final_position_index, 0])
                    
                    for detect_seq in detected_humans: #사람이 전 프레임에 있던 사람인지 확인
                        if final_position_index == detect_seq[0]: #해당 인덱스의 사람이 존재하는지 검사
                            detected_humans.append([final_position_index, [], 0, 0, [], 0, 0, [], 0, 0, [], 0, 0])
                            pred_raw_part(angle_label, detected_humans[-1])

                else: #전 프레임에 같은 사람이 있을때
                    for detect_seq in detected_humans: #사람이 전 프레임에 있던 사람인지 확인
                        if position_index == detect_seq[0]: #해당 인덱스의 사람이 존재하는지 검사
                            position.append([keylist[n][0][0], keylist[n][0][1], position_index, 0])
                            del last_position[last_p_index]
                            pred_raw_part(angle_label, detect_seq)        

        del_list = []         
        for gone in range(len(last_position)):
            last_position[gone][3] += 1
            if last_position[gone][3] > 10:
                del_list.append(last_position[gone])
                

        if len(del_list) > 0:
            for del_position in range(len(del_list)):
                for del_index in detected_humans:
                    if del_index[0] == del_list[del_position]:
                        last_position.remove(del_list[del_position])
                        detected_humans.remove(del_index)
                        break

def pred_raw_part(object_n, human_seq):
    object_n = object_n[:-1]
    seq_length = 30

    actions = ['assult', 'normal']
    
    nan_skip_count = 10 #오차 허용 프레임
    
    for process_count in range(4):
        human_seq[process_count * 3 + 2] = take_not_nan(object_n[process_count * 2], object_n[process_count * 2 + 1], human_seq[process_count * 3 + 1], human_seq[process_count * 3 + 2])
        
    for process_count in range(4):
        if human_seq[process_count * 3 + 2] == 15:
            human_seq[process_count * 3 + 1] = []
            human_seq[process_count * 3 + 3] = 0
            
    for process_count, model in enumerate([right_arm_model, left_arm_model, right_leg_model, left_leg_model]):
        human_seq[process_count * 3 + 3] = pred_part(human_seq[process_count * 3 + 1], human_seq[process_count * 3 + 2], model, human_seq[process_count * 3 + 3])
        
def take_not_nan(key1, key2, part_seq, part_nan):
    if not np.isnan(key1) and not np.isnan(key2):
        part_seq.append([key1, key2])
        return 0
    else:
        part_nan += 1
        if part_nan > 10: #오차 허용 프레임
            return 15
        else:
            return part_nan
        
def pred_part(part_seq, part_nan, part_model, part_action):
    seq_length = 30
    if len(part_seq) > seq_length and part_nan == 0:
        input_data = np.expand_dims(np.array(part_seq[-seq_length:], dtype=np.float32), axis=0)
        y_pred = part_model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf > 0.5 and i_pred == 0: #신뢰도
            part_action += 1
            print(part_action)
            if part_action > 2:
                list_txt.append(('Detect Assult at ', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), ' ',  conf, '%'))
            with open("./human_problem.txt") as f:
                f.writelines(lines)
            
            return part_action
        else:
            return 0
    return part_action

def path_to_filename(fileDir):
    file_name = []
    for name in fileDir:
        file_name.append(ntpath.basename(name))
    return file_name

def video_start(list_file):
#     now = 'test_video.mp4'
#     root_dir = 'C:/Users/chltp/areyouok_pred'
#     video_dir = root_dir + '/datavideo/' + now
    list_txt=[]
    
    for i in list_file:
        video_name = ntpath.basename(name)
        pred_multiple_onvideo(video_path, video_name)

root = tk.Tk()
root.title("Are You OK?")
root.geometry("640x640+100+300")

# 파일추가
def add_file():
    file = filedialog.askopenfilenames(title="동영상 파일을 선택하세요.",
                                       filetypes=(
                                           ("MOV 파일", "*.mov *.mp4"), ("모든 파일", "*.*")),
                                       initialdir="C:/")  # "C:/"
        
    # 사용자가 선택한 파일 목록
    for file in file:
        list_file.insert(END, file)
    

# 선택삭제
def del_file():
    for index in reversed(list_file.curselection()):
        list_file.delete(index)


# log 파일 읽기


# def log(file):
#     if os.path.isfile(file):
#         with open(file, "r", encoding='UTF-8') as f:
#             log_text.insert(END, f.read())



# 시작
def start():
    # 카메라 확인(카메라.파일 둘 중 하나만 가능하도록), log 출력시키기
    video_start(list_file)
    tf = open("./human_problem.txt",
              'r', encoding="UTF-8")
    
    data = tf.read()
    log_text.insert(END, data)
    tf.close()

    # 파일 목록 확인
    if list_file.size() == 0:
        msgbox.showwarning("경고", "동영상 파일을 추가하세요.")
    # elif log_file.size() == 0:
    #     msgbox.showwarning("경고", "로그 파일을 읽는데 문제가 생겼습니다. 다시시도하세요.")
        return

    # 로그 파일 불러오기
    # elif log_file.size() > 0:
    #     for i in range(len(log_file)):
    #         list_log.insert(INSERT, log_file[i])


# cam 선택 (5번까지)
def select_cam():
    i = 0
    cam = cv2.VideoCapture(i)
    while i < 5:
        i = i+1
        if i > 5:
            msgbox.showerror("경고", "내·외장 카메라가 인식되지 않습니다.")
        else:
            msgbox.showinfo("정보", "성공적으로 카메라가 연결되었습니다.")
            break
    
    pred_multiple_onvideo("cam",i)


# (카메라)장치 프레임
cam_frame = Frame(root)
cam_frame.pack(fill="x", padx=5, pady=5)

btn_select_cam = Button(cam_frame, padx=5, pady=5,
                        width=12, text="카메라 선택", command=select_cam)
btn_select_cam.pack(side="left")

# 파일 첨부 프레임
file_frame = Frame(root)
file_frame.pack(fill="x", padx=5, pady=5)

btn_add_file = Button(file_frame, padx=5, pady=5,
                      width=12, text="파일추가", command=add_file)
btn_add_file.pack(side="left")

btn_del_file = Button(file_frame, padx=5, pady=5,
                      width=12, text="선택삭제", command=del_file)
btn_del_file.pack(side="right")


# 파일 리스트 프레임
list_frame = LabelFrame(root, text="파일 경로 목록")
list_frame.pack(fill="both", padx=5, pady=5)

scrollbar = Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")

list_file = Listbox(list_frame, selectmode="extended",
                    height=5, yscrollcommand=scrollbar.set)
list_file.pack(side="left", fill="both", expand=True, padx=5, pady=5)
scrollbar.config(command=list_file.yview)


# 파일 저장 경로 프레임

# 분석값 텍스트 박스 프레임(log값을 반환)
log_frame = LabelFrame(root, text="log값")
log_frame.pack(fill="both", padx=5, pady=5)

# frame_prograss = Frame(root)
# frame_prograss.pack(fill="both", padx=5, pady=5)

scrol_w = 85
scrol_h = 20

log = open("./human_problem.txt", 'r') #, encoding="UTF-8"

log_text = scrolledtext.ScrolledText(log_frame, width=scrol_w, height=scrol_h, wrap=tk.WORD)  # wrap = tk.WORD - 단어단위 줄바꿈  지정
log_text.grid(column=0, columnspan=10)

log_data=log.read()
log_text.insert(tk.INSERT, log_data)
log_text.configure(state="disabled")

# log_text=scrolledtext.ScrolledText(win)
# log_text.config(width=35, height=5, font=("맑은 고딕", 11))
# log_textt.insert(END,"안녕하세요 환영합니다.")
# log_text.configure(state='disabled') #텍스트 위젯을 읽기 전용으로 설정

# 실행 프레임
frame_run = Frame(root)
frame_run.pack(fill="x", padx=5, pady=5)

btn_close = Button(frame_run, padx=5, pady=5, text="닫기",
                   width=12, command=root.destroy)
btn_close.pack(side="right", padx=5, pady=5)

btn_start = Button(frame_run, padx=5, pady=5,
                   text="시작", width=12, command=start)
btn_start.pack(side="right", padx=5, pady=5)

root.resizable(False, False)
root.mainloop()