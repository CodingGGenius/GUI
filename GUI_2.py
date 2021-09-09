import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
import tkinter as tk
import cv2
import datetime
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext


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
    tf = open("/Users/chung/Multi_Pose/human_problem.txt",
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
    global i
    i = 0
    cam = cv2.VideoCapture(i)
    while i < 5:
        i = i+1
        if i > 5:
            msgbox.showerror("경고", "내·외장 카메라가 인식되지 않습니다.")
        else:
            msgbox.showinfo("정보", "성공적으로 카메라가 연결되었습니다.")
            break


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

log = open("/Users/chung/Multi_Pose/human_problem.txt",
           'r')  # , encoding="UTF-8"
log_save = ("/Users/chung/Multi_Pose/log_save.txt", "w")  # 로그값 저장하는 곳

log_text = scrolledtext.ScrolledText(
    log_frame, width=scrol_w, height=scrol_h, wrap=tk.WORD)  # wrap = tk.WORD - 단어단위 줄바꿈  지정
log_text.grid(column=0, columnspan=10)

log_data = log.read()
log_text.insert(tk.INSERT, log_data)
log_text.configure(state='disabled')

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
