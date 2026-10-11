import os
import sys
import csv
from pathlib import Path

import cv2
import PyQt5

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 直接指定 PyQt5 实际安装的插件目录
QT_PLUGIN_PATH = (
    Path(PyQt5.__file__).resolve().parent
    / "Qt5"
    / "plugins"
)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLUGIN_PATH)
os.environ.pop("QT_QPA_FONTDIR", None)

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel,
    QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QMessageBox,
    QFileDialog, QGroupBox, QSpinBox,
    QDateEdit, QCheckBox, QInputDialog
)

from database import ParkingDB
from recognition_adapter import PlateRecognizer
STYLE='''
QWidget {font-family: "Microsoft YaHei", "Noto Sans CJK SC";font-size:13px;color:#25324a;background:#f4f7fb}
QGroupBox {font-weight:bold;border:1px solid #dce4ed;border-radius:10px;margin-top:14px;padding:14px;background:white}
QGroupBox::title {subcontrol-origin:margin;left:16px;padding:0 5px}
QPushButton {background:#2563eb;color:white;padding:9px 16px;border:0;border-radius:7px;font-weight:600}
QPushButton:hover {background:#1d4ed8} QPushButton:disabled {background:#aab4c5}
QLineEdit,QComboBox,QDateEdit {background:white;border:1px solid #cdd7e4;border-radius:6px;padding:7px;min-height:20px}
QTableWidget {background:white;gridline-color:#e5eaf1;alternate-background-color:#f7faff}
QHeaderView::section {background:#e9eff8;padding:8px;border:0;font-weight:bold}
QTabWidget::pane {border:0} QTabBar::tab {padding:12px 20px;background:#e5ebf5;margin-right:3px;border-radius:6px}
QTabBar::tab:selected {background:#2563eb;color:white}
'''

def button(label, callback):
    obj=QPushButton(label);obj.clicked.connect(callback);return obj

def grid_table(columns):
    t=QTableWidget(0,len(columns));t.setHorizontalHeaderLabels(columns)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setAlternatingRowColors(True)
    return t

def fill_table(t,rows):
    t.setRowCount(len(rows))
    for i,row in enumerate(rows):
        for j,val in enumerate(row):
            item=QTableWidgetItem('' if val is None else str(val));item.setTextAlignment(Qt.AlignCenter)
            t.setItem(i,j,item)

class Login(QWidget):
    def __init__(self,db):
        super().__init__();self.db=db;self.dashboard=None
        self.setWindowTitle('停车管理系统 · 登录');self.resize(420,290)
        layout=QVBoxLayout(self);layout.setContentsMargins(44,28,44,30)
        heading=QLabel('智能车牌识别与停车场管理系统');heading.setStyleSheet('font-size:19px;font-weight:bold;color:#1e40af')
        layout.addWidget(heading);form=QFormLayout();self.user=QLineEdit('admin')
        self.password=QLineEdit();self.password.setEchoMode(QLineEdit.Password)
        self.password.returnPressed.connect(self.try_login)
        form.addRow('用户名',self.user);form.addRow('密码',self.password);layout.addLayout(form)
        layout.addWidget(button('登 录',self.try_login));layout.addWidget(QLabel('首次使用：管理员 admin / admin123（登录后请修改默认密码）'))

    def try_login(self):
        role=self.db.login(self.user.text(),self.password.text())
        if not role: QMessageBox.warning(self,'登录失败','账号或密码错误');return
        self.dashboard=Dashboard(self.db,role['role']);self.dashboard.show();self.close()

class Dashboard(QMainWindow):
    def __init__(self,db,role):
        super().__init__();self.db=db;self.role=role;self.recognizer=PlateRecognizer()
        self.camera=None;self.last_frame=None;self.timer=QTimer(self);self.timer.timeout.connect(self.camera_tick)
        self.setWindowTitle('智能车牌识别与停车场管理系统 · '+ ('管理员' if role=='admin' else '操作员'))
        self.resize(1250,800);self.tabs=QTabWidget();self.setCentralWidget(self.tabs)
        self.make_dashboard();self.make_recognition();self.make_records();self.make_blacklist();self.make_members()
        self.refresh_all();self.statusBar().showMessage('系统就绪。识别模板状态：'+('已加载' if self.recognizer.templates else '缺失（可使用手动输入演示）'))

    def notify(self,message): QMessageBox.information(self,'操作结果',message)
    def warn(self,error): QMessageBox.warning(self,'提示',str(error))

    def make_dashboard(self):
        pane=QWidget();root=QVBoxLayout(pane)
        headline=QLabel('停车场运行概览');headline.setStyleSheet('font-size:24px;font-weight:bold;color:#193b78')
        root.addWidget(headline)
        self.stats_label=QLabel();self.stats_label.setStyleSheet('font-size:20px;font-weight:bold;padding:18px;background:white;border-radius:12px')
        root.addWidget(self.stats_label)
        root.addWidget(QLabel('在场车辆 / 车位状态'))
        self.active_table=grid_table(['车牌','车型','车位','入场时间'])
        root.addWidget(self.active_table);root.addWidget(button('刷新统计',self.refresh_all))
        self.tabs.addTab(pane,'首页总览')

    def make_recognition(self):
        pane=QWidget();root=QHBoxLayout(pane)
        left=QVBoxLayout();right=QVBoxLayout()
        self.preview=QLabel('选择车牌图片，或启动摄像头');self.preview.setMinimumSize(610,410)
        self.preview.setAlignment(Qt.AlignCenter);self.preview.setStyleSheet('background:#15233c;color:white;border-radius:12px')
        left.addWidget(self.preview)
        actions=QHBoxLayout();actions.addWidget(button('打开图片识别',self.open_image));actions.addWidget(button('启动摄像头',self.start_camera));actions.addWidget(button('识别当前画面',self.capture_recognize));actions.addWidget(button('关闭摄像头',self.stop_camera));left.addLayout(actions)
        hint=QLabel('当前后端主要支持普通蓝底七位车牌；识别失败可人工纠正车牌号。')
        hint.setWordWrap(True);left.addWidget(hint);left.addStretch();root.addLayout(left,3)
        box=QGroupBox('车辆业务处理');form=QFormLayout(box)
        self.plate=QLineEdit();self.plate.setPlaceholderText('识别后自动填写，也可手动输入')
        self.vehicle=QComboBox();self.vehicle.addItems(['小型车','大型车'])
        self.recognition_state=QLabel('等待识别');self.recognition_state.setWordWrap(True)
        form.addRow('识别状态',self.recognition_state);form.addRow('车牌号',self.plate);form.addRow('车辆类型',self.vehicle)
        form.addRow(button('车辆入场登记',self.enter_car));form.addRow(button('查看出场费用',self.quote_car));form.addRow(button('确认出场结算',self.exit_car))
        right.addWidget(box);right.addStretch();root.addLayout(right,1)
        self.tabs.addTab(pane,'车牌识别 / 进出场')

    def make_records(self):
        pane=QWidget();root=QVBoxLayout(pane);filterbar=QHBoxLayout()
        self.search_plate=QLineEdit();self.search_plate.setPlaceholderText('输入车牌关键字')
        self.start_date=QDateEdit();self.end_date=QDateEdit();self.start_date.setCalendarPopup(True);self.end_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat('yyyy-MM-dd');self.end_date.setDisplayFormat('yyyy-MM-dd')
        self.start_date.setDate(self.start_date.minimumDate());self.end_date.setDate(self.end_date.maximumDate())
        self.only_active=QCheckBox('只看在场')
        filterbar.addWidget(QLabel('车牌'));filterbar.addWidget(self.search_plate)
        filterbar.addWidget(QLabel('开始'));filterbar.addWidget(self.start_date)
        filterbar.addWidget(QLabel('结束'));filterbar.addWidget(self.end_date)
        filterbar.addWidget(self.only_active);filterbar.addWidget(button('查询',self.refresh_records))
        root.addLayout(filterbar)
        self.records_table=grid_table(['ID','车牌','车型','车位','入场时间','出场时间','计费小时','实收费用'])
        root.addWidget(self.records_table)
        controls=QHBoxLayout();controls.addWidget(button('导出当前查询 CSV',self.export_csv))
        if self.role=='admin':
            controls.addWidget(button('修改选中记录车型',self.edit_selected))
            controls.addWidget(button('删除选中记录',self.delete_selected))
        controls.addStretch();root.addLayout(controls)
        self.tabs.addTab(pane,'停车记录查询')

    def make_blacklist(self):
        pane=QWidget();root=QVBoxLayout(pane)
        self.black_table=grid_table(['车牌号','加入原因','加入时间']);root.addWidget(self.black_table)
        bar=QHBoxLayout();self.black_plate=QLineEdit();self.black_plate.setPlaceholderText('车牌')
        self.black_reason=QLineEdit();self.black_reason.setPlaceholderText('加入原因')
        bar.addWidget(self.black_plate);bar.addWidget(self.black_reason)
        if self.role=='admin':
            bar.addWidget(button('添加 / 更新',self.add_black));bar.addWidget(button('删除选中项',self.remove_black))
        root.addLayout(bar);self.tabs.addTab(pane,'黑名单')

    def make_members(self):
        pane=QWidget();root=QVBoxLayout(pane)
        root.addWidget(QLabel('折扣系数：1.0 为原价；0.8 为八折；0 为免费。仅在有效期内享受折扣。'))
        self.member_table=grid_table(['车牌','会员到期日','费用系数']);root.addWidget(self.member_table)
        bar=QHBoxLayout();self.member_plate=QLineEdit();self.member_plate.setPlaceholderText('车牌')
        self.member_until=QDateEdit();self.member_until.setCalendarPopup(True);self.member_until.setDate(self.member_until.date().addYears(1))
        self.member_until.setDisplayFormat('yyyy-MM-dd')
        self.member_discount=QComboBox();self.member_discount.addItems(['1.0','0.9','0.8','0.5','0.0'])
        bar.addWidget(self.member_plate);bar.addWidget(QLabel('有效期'));bar.addWidget(self.member_until)
        bar.addWidget(QLabel('收费系数'));bar.addWidget(self.member_discount)
        if self.role=='admin':
            bar.addWidget(button('添加 / 更新',self.add_member));bar.addWidget(button('删除选中项',self.remove_member))
        root.addLayout(bar);self.tabs.addTab(pane,'会员管理')

    def refresh_all(self):
        total,occupied,free,revenue=self.db.stats()
        self.stats_label.setText(f'总车位  {total}     已占用  {occupied}     剩余  {free}     累计收入  ¥{revenue:.2f}')
        fill_table(self.active_table,[(r['plate'],r['vehicle_type'],r['space_code'],r['entry_time']) for r in self.db.records(active_only=True)])
        self.refresh_records();fill_table(self.black_table,[(r['plate'],r['reason'],r['created_at']) for r in self.db.list_blacklist()])
        fill_table(self.member_table,[(r['plate'],r['valid_until'],r['discount']) for r in self.db.list_members()])

    def refresh_records(self):
        start=self.start_date.date().toString('yyyy-MM-dd') if self.start_date.date()>self.start_date.minimumDate() else ''
        end=self.end_date.date().toString('yyyy-MM-dd') if self.end_date.date()<self.end_date.maximumDate() else ''
        self.current_records=self.db.records(self.search_plate.text(),start,end,self.only_active.isChecked())
        fill_table(self.records_table,[(r['id'],r['plate'],r['vehicle_type'],r['space_code'],r['entry_time'],r['exit_time'],r['duration_hours'],r['fee']) for r in self.current_records])

    def plate_value(self):
        value=self.plate.text().strip().upper()
        if not value: raise ValueError('请输入或识别车牌号')
        return value

    def enter_car(self):
        try:
            code,time=self.db.enter(self.plate_value(),self.vehicle.currentText())
            self.notify(f'车辆入场登记成功\n车位：{code}\n入场时间：{time}');self.refresh_all()
        except Exception as e:self.warn(e)

    def quote_car(self):
        try:
            rec,hours,fee,discount=self.db.quote(self.plate_value())
            self.notify(f'车牌：{rec["plate"]}\n车位：{rec["space_code"]}\n计费时长：{hours} 小时\n应收费用：¥{fee:.2f}')
        except Exception as e:self.warn(e)

    def exit_car(self):
        try:
            plate=self.plate_value();rec,hours,fee,discount=self.db.quote(plate)
            answer=QMessageBox.question(self,'确认结算',f'{plate}\n停放 {hours} 小时，应收 ¥{fee:.2f}\n确认收款并放行？')
            if answer!=QMessageBox.Yes:return
            code,hours,fee=self.db.exit(plate)
            self.notify(f'结算成功：¥{fee:.2f}\n车位 {code} 已释放');self.refresh_all()
        except Exception as e:self.warn(e)

    def render_frame(self,frame):
        if frame is None:return
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);h,w,c=rgb.shape
        img=QImage(rgb.data,w,h,c*w,QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def recognize_frame(self,frame):
        try:
            value,box=self.recognizer.recognize_frame(frame)
            if box:
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),3)
            self.render_frame(frame)
            if value:
                self.plate.setText(value);self.recognition_state.setText('识别成功：'+value)
                if self.db.is_blacklisted(value):self.warn('注意：识别到黑名单车辆 '+value)
            else:self.recognition_state.setText('未检测到有效车牌；请换图或人工输入')
        except Exception as e:self.recognition_state.setText('识别失败：'+str(e));self.warn(e)

    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,'选择车牌照片',str(Path.home()),'图片 (*.jpg *.jpeg *.png *.bmp)')
        if not path:return
        self.stop_camera()
        try:
            import numpy as np
            frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:raise ValueError('无法读取该图片')
            self.last_frame=frame.copy();self.recognize_frame(frame)
        except Exception as e:self.warn(e)

    def start_camera(self):
        self.stop_camera();self.camera=cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release();self.camera=None;self.warn('无法打开摄像头，请检查设备或权限');return
        self.timer.start(80);self.recognition_state.setText('摄像头画面显示中；点击「识别当前画面」')

    def camera_tick(self):
        if self.camera:
            ok,frame=self.camera.read()
            if ok:self.last_frame=frame.copy();self.render_frame(frame)

    def capture_recognize(self):
        if self.last_frame is None:self.warn('没有可供识别的画面');return
        self.recognize_frame(self.last_frame.copy())

    def stop_camera(self):
        self.timer.stop()
        if self.camera:self.camera.release();self.camera=None

    def selected_id(self):
        row=self.records_table.currentRow()
        if row<0:raise ValueError('请先选择一条停车记录')
        return int(self.records_table.item(row,0).text())

    def delete_selected(self):
        try:
            ident=self.selected_id()
            if QMessageBox.question(self,'确认删除',f'确认删除记录 #{ident}？此操作无法撤销。')==QMessageBox.Yes:
                self.db.delete_record(ident);self.refresh_all()
        except Exception as e:self.warn(e)

    def edit_selected(self):
        try:
            ident=self.selected_id()
            kind,ok=QInputDialog.getItem(self,'修改车型','选择车型',['小型车','大型车'],0,False)
            if ok:self.db.change_vehicle_type(ident,kind);self.refresh_all()
        except Exception as e:self.warn(e)

    def export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,'导出停车记录','parking_records.csv','CSV 文件 (*.csv)')
        if not path:return
        try:
            with open(path,'w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['ID','车牌','车型','车位','入场时间','出场时间','计费小时','实收费用'])
                for r in self.current_records:
                    writer.writerow([r[k] for k in ('id','plate','vehicle_type','space_code','entry_time','exit_time','duration_hours','fee')])
            self.notify('已导出 '+path)
        except Exception as e:self.warn(e)

    def add_black(self):
        try:
            plate=self.black_plate.text().strip().upper()
            if not plate:raise ValueError('车牌不能为空')
            self.db.add_blacklist(plate,self.black_reason.text());self.refresh_all()
        except Exception as e:self.warn(e)

    def remove_black(self):
        row=self.black_table.currentRow()
        if row<0:self.warn('请选中黑名单记录');return
        self.db.remove_blacklist(self.black_table.item(row,0).text());self.refresh_all()

    def add_member(self):
        try:
            plate=self.member_plate.text().strip().upper()
            if not plate:raise ValueError('车牌不能为空')
            self.db.add_member(plate,self.member_until.date().toString('yyyy-MM-dd'),float(self.member_discount.currentText()))
            self.refresh_all()
        except Exception as e:self.warn(e)

    def remove_member(self):
        row=self.member_table.currentRow()
        if row<0:self.warn('请选中会员记录');return
        self.db.remove_member(self.member_table.item(row,0).text());self.refresh_all()

    def closeEvent(self,event):
        self.stop_camera();event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);app.setStyleSheet(STYLE)
    try:
        window=Login(ParkingDB());window.show();sys.exit(app.exec_())
    except Exception as exc:
        QMessageBox.critical(None,'启动失败',str(exc));raise
