# Requirement  
opencv  5.0.0  
numpy   2.4.6  
matplotlib  3.11.1  
python  3.11.16  
nbconvert  7.17.1


# Execution  
若需要更改圖片路徑，在/Code/quiz1/quiz_1.ipynb裡，第一個code cell有以下標記：  
```python
'''=====圖片路徑更改====='''
image_path = "../Data/quiz1/Zhuang.png"
```  
需要儲存結果，在最後一個cell有：  
```python
'''=====儲存灰階影像路徑====='''
cv2.imwrite("gray_numpy.jpg", gray_numpy)
cv2.imwrite("gray_opencv.jpg", gray_opencv)
```  
其他quiz同理  

之後執行main.py **<某某資料夾\MMIP\week1>python main.py**  
會跳出請輸入要執行的題目（例如 quiz_1）：  
直接輸入quiz_1、quiz_2、quiz_3、quiz_4，等待即可

