# For our experiements, we save the commands to a mysql database, and have computers 
# call the database when they need a new job 
# There is a file called secret.py which houses 
# PYTHONANYWHERE_PASSWORD, PYTHONANYWHERE_USERNAME, DATABASE_HOSTNAME, DATABASE_NAME, DATABASE_PASSWORD
# and we don't want these saved to the git so I have added secret.py to .gitignore
import sys 
import os
import pdb
from secret import *
import pymysql
import sshtunnel
import platform 

try:
    import sh
except ImportError:
    # fallback: emulate the sh API with pbs
    import pbs
    class Sh(object):
        def __getattr__(self, attr):
            return pbs.Command(attr)
    sh = Sh()

sshtunnel.SSH_TIMEOUT = 15.0
sshtunnel.TUNNEL_TIMEOUT = 15.0

# This portion of the script is specific to my home computing setup.
if platform.system() == 'Windows':
    pySRURGS_dir = 'C:/Users/sohra/Google Drive (fischerproject2018@gmail.com)/pySRURGS'
elif platform.system() == 'Linux':
    pySRURGS_dir = '/home/brain/pySRURGS'
else:
    raise Exception("Invalid OS")

def int_finished_meaning(my_int):
    # this integer we use when assigning job completion status to the listing of 
    # jobs in our database 
    int_dict = {0: 'not_run', 1: 'running', 2:'finished'}
    return int_dict[my_int]

def submit_job_to_db(algo_argu_list):
    num_job = len(algo_argu_list)
    with sshtunnel.SSHTunnelForwarder(('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = pymysql.connect(user=PYTHONANYWHERE_USERNAME, 
                                                 password=DATABASE_PASSWORD,
                                                 host='127.0.0.1', 
                                                 port=tunnel.local_bind_port,
                                                 database=DATABASE_NAME)
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
            sql = "INSERT INTO jobs (algorithm, arguments, finished) VALUES (%s, %s, %s)"
            val = (algorithm, arguments, 0)
            mycursor.execute(sql, val)        
        mydb.commit()
        mydb.close()

def purge_db():
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = pymysql.connect(user=PYTHONANYWHERE_USERNAME, 
                                             password=DATABASE_PASSWORD,
                                             host='127.0.0.1', 
                                             port=tunnel.local_bind_port,
                                             database=DATABASE_NAME)
        mycursor = mydb.cursor()                               
        sql = "DROP TABLE jobs'"
        mycursor.execute(sql)
        mydb.commit()
        mydb.close()
    
def get_SRGP_job(finished=0):
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = pymysql.connect(user=PYTHONANYWHERE_USERNAME, 
                                             password=DATABASE_PASSWORD,
                                             host='127.0.0.1', 
                                             port=tunnel.local_bind_port,
                                             database=DATABASE_NAME)
        mycursor = mydb.cursor()                               
        sql = '''SELECT job_ID, arguments FROM jobs
                WHERE finished = %s
                ORDER BY RAND()
                LIMIT 1'''
        val = (finished,)
        mycursor.execute(sql, val)
        myresult = mycursor.fetchone()
        if myresult is None:
            return None
        job_ID = myresult[0]
        arguments = myresult[1]      
        sql = "UPDATE jobs SET finished = 1 WHERE job_ID = %s"
        val = (job_ID,)
        mycursor.execute(sql, val)
        mydb.commit()
        mydb.close()            
    if ';' in arguments:
        raise Exception("SQL insertion? - don't run this.")
    arguments = arguments.split()
    for i in range(0,len(arguments)):
        arguments[i] = arguments[i].replace('$PYSRURGSDIR',pySRURGS_dir)    
    return job_ID, arguments

def set_SRGP_job_finished(job_ID):
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username=PYTHONANYWHERE_USERNAME, ssh_password=PYTHONANYWHERE_PASSWORD,
        remote_bind_address=(DATABASE_HOSTNAME, 3306)) as tunnel:
        mydb = pymysql.connect(user=PYTHONANYWHERE_USERNAME, 
                                             password=DATABASE_PASSWORD,
                                             host='127.0.0.1', 
                                             port=tunnel.local_bind_port,
                                             database=DATABASE_NAME)
        mycursor = mydb.cursor()                               
        sql = "UPDATE jobs SET finished = 2 WHERE job_ID = %s"
        val = (job_ID,)
        mycursor.execute(sql, val)
        mydb.commit()
        mydb.close()            
    
def run_all_SRGP_jobs():
    i = 0
    for finished in range(0,2):
        job_ID, job_arguments = get_SRGP_job(finished)
        while job_arguments is not None:
            if ((job_arguments[0] != '-m') 
               or (job_arguments[1] != 'scoop') 
               or (job_arguments[2] != pySRURGS_dir+'/experiments/SRGP.py')):
                raise Exception("SQL injection?")
            try:
                sh.python(*job_arguments)  
                sh.git('pull')
                sh.git('add', job_arguments[4])
                sh.git('commit', '-m', os.path.basename(job_arguments[4]), job_arguments[4])
                sh.git('push')                
            except sh.ErrorReturnCode as e:
                print(e.stderr)
            set_SRGP_job_finished(job_ID)
            job_ID, job_arguments = get_SRGP_job(finished)
            print('finished a job', i)
            i = i + 1

if __name__ == '__main__':
    run_all_SRGP_jobs()
