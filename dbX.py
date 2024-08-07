import os
from dotenv import load_dotenv
import mysql.connector

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),              
            user=os.getenv('DB_USER'),                   
            password=os.getenv('DB_PASSWORD'),     
            database=os.getenv('DB_NAME')               
        )
        self.cursor = self.connection.cursor(dictionary=True)  # Use dictionary=True to fetch data as dictionaries

    def get_database_connection(self):
        if not self.connection or not self.cursor:
            self.connect()
        return self.cursor

    def fetch_drug_data(self):
        self.cursor.execute("SELECT * FROM Drugs")
        return self.cursor.fetchall()

    def fetch_medical_tests_data(self):
        self.cursor.execute("SELECT * FROM MedicalTests")
        return self.cursor.fetchall()

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
