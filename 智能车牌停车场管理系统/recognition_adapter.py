"""将原始识别脚本包装为供图形界面调用的接口，不更改识别算法。"""
import re
import cv2
import numpy as np
import plate_recognition as legacy
from pathlib import Path

BASE=Path(__file__).resolve().parent

class PlateRecognizer:
    def __init__(self):
        self.templates=None
        self.error=None
        root=BASE / 'plate_template_24x48_done'
        try:
            self.templates=legacy.load_templates(str(root))
            # Windows 打包保存的中文文件名可能为 #U4e91 形式，恢复为 Unicode 字符。
            chinese=self.templates['chinese']
            self.templates['chinese']={
                (chr(int(k[2:],16)) if k.startswith('#U') and len(k)>2 else k):v
                for k,v in chinese.items()
            }
            if any(not self.templates[k] for k in ('chinese','letter','number')):
                raise ValueError('汉字、字母、数字模板均不能为空')
        except (FileNotFoundError,ValueError) as exc:
            self.error=str(exc)

    def recognize_frame(self, frame):
        if self.templates is None:
            raise RuntimeError('车牌识别模板未就绪：'+str(self.error))
        plate,mask,box=legacy.locate_blue_plate(frame)
        if plate is None:
            return '',None
        binary=legacy.plate_binarization(plate)
        chars,roi,regions=legacy.segment_characters(binary)
        value=legacy.recognize_plate(chars,self.templates)
        if not (isinstance(value,str) and len(value)==7 and re.fullmatch(r'[\u4e00-\u9fff][A-Z][A-Z0-9]{5}',value)):
            return '',box
        return value,box

    def recognize_file(self,path):
        frame=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None: raise ValueError('无法读取图片')
        value,box=self.recognize_frame(frame)
        if box:
            x1,y1,x2,y2=box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(32,200,100),3)
        return value,frame
