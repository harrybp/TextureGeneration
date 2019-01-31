import sqlite3

def init_db():
    query_database('CREATE TABLE gan (name TEXT, iterations INT, lr FLOAT, size INT)')
    query_database('CREATE TABLE progress (subject TEXT, progress INT)')
    print("Tables created successfully")

def clear_db():
    query_database('DROP TABLE IF EXISTS gan')
    query_database('DROP TABLE IF EXISTS progress')
    print("Tables deleted successfully")

def populate_db():
    query_database('INSERT INTO progress (subject, progress) VALUES ("gatys", 0)')
    query_database('INSERT INTO progress (subject, progress) VALUES ("gan", 0)')
    save_GAN('bricks', 254, 0.00002, 256)
    save_GAN('pebbles', 254, 0.00002, 256)
    save_GAN('snake', 254, 0.00002, 256)
    save_GAN('water', 254, 0.00002, 256)


def list_db():
    gan = query_database('SELECT * FROM gan')
    progress = query_database('SELECT * FROM progress')
    print('TABLE: gan')
    for row in gan:
        print(''.join((str(r) + ' ') for r in row))
    print('TABLE: PROGRESS')
    for row in progress:
        print(''.join((str(r) + ' ') for r in row))


def query_database(query):
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(query)
            results = cur.fetchall()
            con.commit()
            msg = None
    except:
        results = []
        con.rollback()
        msg = "Database query failure"
    finally:
        if msg:
            print(msg)
        con.close()
        return results

#Add a GAN to the database
def save_GAN(name, iterations, lr, size):
    query = 'INSERT INTO gan (name,iterations,lr,size) VALUES ("' + name + '",' + str(iterations) + ',' + str(lr) + ',' + str(size) + ')'
    query_database(query)

def update_GAN(name, iterations):
    query = 'UPDATE gan SET iterations = ' + str(iterations) + ' WHERE name = "'+name+'"'
    query_database(query)

def update_progress(progress, subject):
    query = 'UPDATE progress SET progress = ' + str(progress) + ' WHERE subject = "'+ subject +'"'
    query_database(query)

def get_progress(subject):
    query = 'SELECT progress FROM progress WHERE subject = "'+ subject +'"'
    results = query_database(query)
    return results[0][0]


#clear_db()
#init_db()
#populate_db()
#list_db()