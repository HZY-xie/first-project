import tempfile
import unittest
from datetime import datetime,timedelta
from pathlib import Path
from database import ParkingDB

class DBTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=ParkingDB(Path(self.temp.name)/'test.db')
    def tearDown(self):
        self.db.conn.close();self.temp.cleanup()

    def test_login_and_initial_capacity(self):
        self.assertEqual(self.db.login('admin','admin123')['role'],'admin')
        self.assertIsNone(self.db.login('admin','wrong'))
        self.assertEqual(self.db.stats()[:3],(50,0,50))

    def test_entry_exit_and_duplicate(self):
        t=datetime(2026,10,9,10,0)
        space,_=self.db.enter('粤A12345','小型车',t)
        self.assertEqual(space,'A001')
        with self.assertRaises(ValueError):self.db.enter('粤A12345')
        space,hours,fee=self.db.exit('粤A12345',t+timedelta(minutes=61))
        self.assertEqual((space,hours,fee),('A001',2,4.0))
        self.assertEqual(self.db.stats()[:3],(50,0,50))

    def test_minimum_hour_and_member(self):
        t=datetime(2026,10,9,10)
        self.db.add_member('粤B12345','2026-12-31',0.5)
        self.db.enter('粤B12345','大型车',t)
        _,hours,fee=self.db.exit('粤B12345',t+timedelta(minutes=10))
        self.assertEqual((hours,fee),(1,2.0))

    def test_blacklist(self):
        self.db.add_blacklist('粤C12345','测试限制')
        with self.assertRaises(ValueError):self.db.enter('粤C12345')

    def test_full_capacity(self):
        for i in range(50):self.db.enter(f'粤D{i:05d}')
        self.assertEqual(self.db.stats()[:3],(50,50,0))
        with self.assertRaisesRegex(ValueError,'已满'):self.db.enter('粤E12345')

    def test_admin_delete_active_restores_space(self):
        self.db.enter('粤F12345')
        row=self.db.records(active_only=True)[0]
        self.db.delete_record(row['id'])
        self.assertEqual(self.db.stats()[:3],(50,0,50))

if __name__=='__main__':unittest.main()
