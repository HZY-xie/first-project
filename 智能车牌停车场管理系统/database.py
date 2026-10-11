import hashlib
import math
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / 'parking.db'

def password_hash(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

class ParkingDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.init_db()

    def init_db(self):
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','operator')));
        CREATE TABLE IF NOT EXISTS spaces(id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL,
            occupied INTEGER NOT NULL DEFAULT 0 CHECK(occupied IN (0,1)));
        CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, plate TEXT NOT NULL,
            vehicle_type TEXT NOT NULL, space_code TEXT NOT NULL,
            entry_time TEXT NOT NULL, exit_time TEXT, duration_hours INTEGER,
            fee REAL, member_discount REAL NOT NULL DEFAULT 0);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_plate ON records(plate) WHERE exit_time IS NULL;
        CREATE TABLE IF NOT EXISTS blacklist(plate TEXT PRIMARY KEY, reason TEXT DEFAULT '', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS members(plate TEXT PRIMARY KEY, valid_until TEXT NOT NULL,
            discount REAL NOT NULL DEFAULT 1 CHECK(discount >= 0 AND discount <= 1));
        ''')
        if not self.conn.execute('SELECT 1 FROM users LIMIT 1').fetchone():
            self.conn.execute('INSERT INTO users(username,password_hash,role) VALUES(?,?,?)',
                              ('admin', password_hash('admin123'), 'admin'))
        if not self.conn.execute('SELECT 1 FROM spaces LIMIT 1').fetchone():
            self.conn.executemany('INSERT INTO spaces(code) VALUES(?)', [(f'A{i:03d}',) for i in range(1,51)])
        self.conn.commit()
        self.reconcile_spaces()

    def reconcile_spaces(self):
        with self.conn:
            self.conn.execute('UPDATE spaces SET occupied=0')
            self.conn.execute('''UPDATE spaces SET occupied=1 WHERE code IN
                (SELECT space_code FROM records WHERE exit_time IS NULL)''')

    def login(self, username, password):
        return self.conn.execute('SELECT role FROM users WHERE username=? AND password_hash=?',
                                 (username.strip(),password_hash(password))).fetchone()

    def stats(self):
        total=self.conn.execute('SELECT COUNT(*) FROM spaces').fetchone()[0]
        occupied=self.conn.execute('SELECT COUNT(*) FROM spaces WHERE occupied=1').fetchone()[0]
        revenue=self.conn.execute('SELECT COALESCE(SUM(fee),0) FROM records WHERE exit_time IS NOT NULL').fetchone()[0]
        return total, occupied, total-occupied, revenue

    def is_blacklisted(self, plate):
        return self.conn.execute('SELECT reason FROM blacklist WHERE plate=?',(plate,)).fetchone()

    def enter(self, plate, vehicle_type='小型车', when=None):
        plate=plate.strip().upper()
        if not plate:
            raise ValueError('车牌不能为空')
        black=self.is_blacklisted(plate)
        if black:
            raise ValueError('黑名单车辆禁止入场：'+ (black['reason'] or '未注明原因'))
        now=(when or datetime.now()).isoformat(timespec='seconds')
        with self.conn:
            if self.conn.execute('SELECT 1 FROM records WHERE plate=? AND exit_time IS NULL',(plate,)).fetchone():
                raise ValueError('该车牌已有未出场记录')
            space=self.conn.execute('SELECT code FROM spaces WHERE occupied=0 ORDER BY id LIMIT 1').fetchone()
            if not space:
                raise ValueError('停车场已满，禁止入场')
            self.conn.execute('UPDATE spaces SET occupied=1 WHERE code=?',(space['code'],))
            self.conn.execute('INSERT INTO records(plate,vehicle_type,space_code,entry_time) VALUES(?,?,?,?)',
                              (plate,vehicle_type,space['code'],now))
        return space['code'],now

    def quote(self, plate, when=None):
        plate=plate.strip().upper()
        record=self.conn.execute('SELECT * FROM records WHERE plate=? AND exit_time IS NULL',(plate,)).fetchone()
        if not record:
            raise ValueError('没有查到该车辆的在场记录')
        now=when or datetime.now()
        delta=(now-datetime.fromisoformat(record['entry_time'])).total_seconds()
        if delta < 0:
            raise ValueError('出场时间不能早于入场时间')
        hours=max(1,math.ceil(delta/3600))
        rate=2 if record['vehicle_type']=='小型车' else 4
        member=self.conn.execute('SELECT discount,valid_until FROM members WHERE plate=?',(plate,)).fetchone()
        multiplier=1.0
        if member and member['valid_until']>=now.date().isoformat():
            multiplier=float(member['discount'])
        fee=round(hours*rate*multiplier,2)
        return record,hours,fee,multiplier

    def exit(self, plate, when=None):
        now=when or datetime.now()
        with self.conn:
            record,hours,fee,multiplier=self.quote(plate,now)
            self.conn.execute('''UPDATE records SET exit_time=?,duration_hours=?,fee=?,member_discount=? WHERE id=?''',
                              (now.isoformat(timespec='seconds'),hours,fee,1-multiplier,record['id']))
            self.conn.execute('UPDATE spaces SET occupied=0 WHERE code=?',(record['space_code'],))
        return record['space_code'],hours,fee

    def records(self, plate='',date_from='',date_to='',active_only=False):
        sql='SELECT * FROM records WHERE 1=1'; args=[]
        if plate:
            sql+=' AND plate LIKE ?'; args.append('%'+plate.strip().upper()+'%')
        if date_from:
            sql+=' AND entry_time >= ?'; args.append(date_from+'T00:00:00')
        if date_to:
            sql+=' AND entry_time <= ?'; args.append(date_to+'T23:59:59')
        if active_only:
            sql+=' AND exit_time IS NULL'
        return self.conn.execute(sql+' ORDER BY id DESC',args).fetchall()

    def delete_record(self, record_id):
        with self.conn:
            row=self.conn.execute('SELECT space_code,exit_time FROM records WHERE id=?',(record_id,)).fetchone()
            if not row: raise ValueError('记录不存在')
            self.conn.execute('DELETE FROM records WHERE id=?',(record_id,))
            if row['exit_time'] is None:
                self.conn.execute('UPDATE spaces SET occupied=0 WHERE code=?',(row['space_code'],))

    def change_vehicle_type(self, record_id, kind):
        if kind not in ('小型车','大型车'): raise ValueError('车辆类型无效')
        with self.conn:
            row=self.conn.execute('SELECT exit_time FROM records WHERE id=?',(record_id,)).fetchone()
            if not row: raise ValueError('记录不存在')
            if row['exit_time'] is not None: raise ValueError('已结算记录不可修改车型，以免影响账目')
            self.conn.execute('UPDATE records SET vehicle_type=? WHERE id=?',(kind,record_id))

    def list_blacklist(self):
        return self.conn.execute('SELECT * FROM blacklist ORDER BY created_at DESC').fetchall()

    def add_blacklist(self, plate,reason):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO blacklist VALUES(?,?,?)',
                              (plate.strip().upper(),reason,datetime.now().isoformat(timespec='seconds')))

    def remove_blacklist(self, plate):
        with self.conn: self.conn.execute('DELETE FROM blacklist WHERE plate=?',(plate,))

    def list_members(self):
        return self.conn.execute('SELECT * FROM members ORDER BY plate').fetchall()

    def add_member(self,plate,valid_until,discount):
        if not 0<=float(discount)<=1: raise ValueError('折扣系数必须在0和1之间')
        datetime.strptime(valid_until,'%Y-%m-%d')
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO members VALUES(?,?,?)',
                              (plate.strip().upper(),valid_until,float(discount)))

    def remove_member(self,plate):
        with self.conn: self.conn.execute('DELETE FROM members WHERE plate=?',(plate,))
