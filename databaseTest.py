from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import g
import sqlite3
import utils


def init_db():
    conn = sqlite3.connect('database.db')
    #print("Opened database successfully")
    #conn.execute('CREATE TABLE gans (name TEXT, sourceimg TEXT, iterations INT, lr FLOAT, width INT, height INT)')
    #print("Table created successfully")
    #conn.execute('CREATE TABLE gatys (current TEXT, progress INT)')
    conn.execute('CREATE TABLE gans_progress (current TEXT, progress INT)')
    print("Table created successfully")
    conn.close()

def clear_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    #conn.execute('DROP TABLE gans')
    conn.execute('DROP TABLE gans_progress')
    print("Table deleted successfully")
    conn.close()


def make_list():
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    
    query = 'SELECT name, MAX(iterations) FROM gans GROUP BY name'
    cur.execute(query)
    rows = cur.fetchall()
    # print((rows[0]['name']))
    print(len(rows))
    for row in rows:
        print(''.join((str(r) + ' ') for r in row))

    cur = con.cursor()
    
    query = 'SELECT * FROM gans_progress'
    cur.execute(query)
    rows = cur.fetchall()
    # print((rows[0]['name']))
    print(len(rows))
    for row in rows:
        print(''.join((str(r) + ' ') for r in row))
    #print(' '.join( [ (r[0] + ' ' + r[1] + ' ' + str(r[2]) + ' ' + str(r[3]) + ' ' + str(r[4]) + ' ' + str(r[5]) + '\n') for r in rows]))


def addrec(iters):
    try:
        name = 'testgadn'
        source = '/ssd/sda/ksd.jpg'
        iterations = iters
        lr = 0.0002
        width = 128
        height = 128

        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            print('ok')
            cur.execute("INSERT INTO gans (name,sourceimg,iterations,lr,width,height) VALUES (?,?,?,?,?,?)",
                        (name, source, iterations, lr, width, height))
            print('na')
            con.commit()
            msg = "Record successfully added"
    except:
        con.rollback()
        msg = "error in insert operation"

    finally:
        print(msg)
        con.close()

def addrec2():
    try:


        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            print('ok')
            cur.execute("INSERT INTO gans_progress (current,progress) VALUES (?,?)",('gans', 0))
            print('na')
            con.commit()
            msg = "Record successfully added"
    except:
        con.rollback()
        msg = "error in insert operation"

    finally:
        print(msg)
        con.close()

def remove_clouds():
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute('DELETE FROM gans WHERE name="camo"')
            con.commit()
            msg = "Records successfully removed"
    except:
        con.rollback()
        msg = "error in remove operation"

    finally:
        print(msg)
        con.close()

def update_rows():
    query = 'UPDATE gans SET name = "snake" WHERE name = "honeycomb"'
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(query)
            con.commit()
            msg = "Records successfully removed"
    except:
        con.rollback()
        msg = "error in remove operation"

    finally:
        print(msg)
        con.close()

#clear_db()
#init_db()
#init_db()
#addrec(1)
#addrec(2)
#addrec(3)
#import json
#make_list()
#addrec2()
#remove_clouds()
#update_rows()
#clear_db()
#init_db()
#addrec2()
make_list()
#utils.update_progress(0)
#make_list()
#utils.get_progress()
#init_db()
#addrec2()
utils.get_progress_gan()
#xo = utils.get_GANS()
#print(json.dumps(xo))