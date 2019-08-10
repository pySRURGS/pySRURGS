# For our experiements, we save the commands to a mysql database, and have computers 
# call the database when they need a new job 
import mysql.connector
import sys 
from secret import *
import mysql.connector
import sshtunnel

sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0

def submit_job_to_db(algo_argu_list):
    num_job = len(algo_argu_list)
    with sshtunnel.SSHTunnelForwarder(('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = mysql.connector.connect(user=PYTHONANYWHERE_USERNAME, 
                                                 password=DATABASE_PASSWORD,
                                                 host='127.0.0.1', 
                                                 port=tunnel.local_bind_port,
                                                 database='SohrabT$pySRURGS')
        i = 0
        for job in algo_argu_list:            
            print(i, num_job)
            i = i + 1
            algorithm = job[0]
            arguments = job[1]
            create_db_command = '''CREATE TABLE IF NOT EXISTS jobs(
               job_ID INT NOT NULL AUTO_INCREMENT,
               algorithm VARCHAR(200) NOT NULL,
               arguments VARCHAR(200) NOT NULL,
               finished TINYINT(1) NOT NULL,
               PRIMARY KEY ( job_ID )
            );'''
            mycursor = mydb.cursor()
            mycursor.execute(create_db_command)
            sql = "INSERT INTO jobs (algorithm, arguments, finished) VALUES (%s, %s, %i)"
            val = (algorithm, arguments, 0)
            mycursor.execute(sql, val)        
        mydb.commit()
        mydb.close()

def purge_db(mydb, username, password):
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = mysql.connector.connect(user=PYTHONANYWHERE_USERNAME, 
                                             password=DATABASE_PASSWORD,
                                             host='127.0.0.1', 
                                             port=tunnel.local_bind_port,
                                             database='SohrabT$pySRURGS')
        mycursor = mydb.cursor()                               
        sql = "DROP TABLE jobs'"
        mycursor.execute(sql)
        mydb.commit()
        mydb.close()