import streamlit as st
import sqlite3 as sqlite

# Connect to SQLite database

conn = sqlite3.connect('chatdb.db')
cursor = conn.cursor()

# Create table if not exists

cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 age INTEGER NOT NULL,
                 department TEXT NOT NULL)''')

