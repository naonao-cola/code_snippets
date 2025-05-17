

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QTableView
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from PyQt5.QtCore import Qt, QTime
import os
import sys

sys.path.append(".")
from utils.ulog import ulog
from utils.image_func import *

class DataEdit():
    def __init__(self, path:str =None):
        self.db = None
        self.db_connect()
        self.path_vec = []
        if path:
            self.path_vec.append(path)
        self.createTable()

    def db_connect(self):
        self.db = QSqlDatabase.addDatabase('QSQLITE')
        current_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.db.setDatabaseName('filesystem.db')
        if not self.db.open():
            QMessageBox.critical(self, 'Database Connection',
                                 self.db.lastError().text())

    def closeEvent(self):
        self.db.close()

    def createTable(self):
        query = QSqlQuery()
        ret = query.exec("CREATE TABLE IF NOT EXISTS filesystem_item ("
                         "sn VARCHAR(24) NOT NULL,"
                         "date VARCHAR(12) NOT NULL,"
                         "sample_id VARCHAR(12) NOT NULL,"
                         "categorize VARCHAR(12) NOT NULL,"
                         "path VARCHAR(256) NOT NULL,"
                         "PRIMARY KEY (sn, date, sample_id,categorize))")
        if not ret:
            ulog().get_logger().error(
                f"SQL 建表 错误, {query.lastError().text()}")

    def add_path(self,path:str):
        if path:
            self.path_vec.append(str(self.path_vec))

    def auto_add(self):
        if self.path_vec:
            for item in self.path_vec:
                image_vec = get_all_folders(item)

        file_data = process_image_vec(image_vec)
        query = QSqlQuery()
        self.db.transaction()
        timer = QTime()
        print(f"字典的大小 {len(file_data)}")
        timer.start()
        for data_item in file_data:
            content = """
            INSERT INTO filesystem_item (sn, date, sample_id, categorize,path) VALUES (?, ?, ?, ?,?)
            ON CONFLICT(sn, date, sample_id, categorize) DO NOTHING
            """
            query.prepare(content)
            query.addBindValue(data_item.get("sn"))
            query.addBindValue(data_item.get("date"))
            query.addBindValue(data_item.get("sample_id"))
            query.addBindValue(data_item.get("categorize"))
            query.addBindValue(data_item.get("path"))
            insert_ret = query.exec()
            if not insert_ret:
                ulog().get_logger().error(
                    f"SQL 插入错误：, {query.lastError().text()}")
                self.db.rollback()

        self.db.commit()
        ulog().get_logger().info(
            f"SQL 插入时间耗时 ：, {timer.elapsed()} ms")

    def auto_sea(self, date, sample_id, categorize):
        query = QSqlQuery()
        query.prepare(
            "SELECT * FROM filesystem_item WHERE date = ? AND sample_id = ? AND categorize = ?")
        query.addBindValue(date)
        query.addBindValue(sample_id)
        query.addBindValue(categorize)
        content = f" SELECT * FROM filesystem_item WHERE date = {date} AND sample_id = {sample_id} AND categorize = {categorize}"
        print(f"查询语句： {content}")
        if not query.exec():
            ulog().get_logger().error(
                f"SQL 查询失败：, {query.lastError().text()}")
            return None
        result = []
        while query.next():
            sn_result = query.value(0)
            date_result = query.value(1)
            sample_id = query.value(2)
            categorize = query.value(3)
            path = query.value(4)
            print(
                f"SN: {sn_result}, Date: {date_result}, Sample ID: {sample_id}, Categorize: {categorize}, Image Name: {path}")
            ulog().get_logger().info(
                f"SQL 查询结果, SN: {sn_result}, Date: {date_result}, Sample ID: {sample_id}, Categorize: {categorize}")
            result.append({"sn": sn_result, "date": date_result,
                          "sample_id": sample_id, "categorize": categorize, "path": path})
        return result

    def auto_sea_all(self):
        query = QSqlQuery()
        query.prepare("SELECT * FROM filesystem_item")

        if not query.exec():
            ulog().get_logger().error(f"SQL 查询失败：, {query.lastError().text()}")
            return None
        result = []
        while query.next():
            sn_result = query.value(0)
            date_result = query.value(1)
            sample_id = query.value(2)
            categorize = query.value(3)
            path = query.value(4)
            print(
                f"SN: {sn_result}, Date: {date_result}, Sample ID: {sample_id}, Categorize: {categorize}, Image Name:  {path}")
            # ulog().get_logger().info(
            #     f"SQL 查询结果, SN: {sn_result}, Date: {date_result}, Sample ID: {sample_id}, Categorize: {categorize}")
            result.append({"sn": sn_result, "date": date_result,
                          "sample_id": sample_id, "categorize": categorize,  "path": path})
        return result

    def auto_del(self,path):
        """只支持删除SN目录
        Args:
            path (_type_): _description_
        """
        # F:\data\SN
        sn_name = os.path.basename(path)
        query = QSqlQuery()
        delete_sql = "DELETE FROM filesystem_item WHERE SN LIKE ?"
        query.prepare(delete_sql)
        query.addBindValue(f"{sn_name}%")
        if not query.exec():
            ulog().get_logger().error(f"删除失败: {query.lastError().text()}")
        else:
            ulog().get_logger().info(f"删除成功: {path}")

if __name__ == "__main__":
    dataobj = DataEdit(r"F:\data\SN")
    dataobj.db_connect()
    dataobj.createTable()
    dataobj.auto_add()
    dataobj.auto_sea(date="20250501", sample_id="0001", categorize="cbc")
    dataobj.auto_sea_all()
    dataobj.closeEvent()
    pass
