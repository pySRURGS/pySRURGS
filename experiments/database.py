# For our experiements, we save the commands to a mysql database, and have computers 
# call the database when they need a new job 
# There is a file called secret.py which houses 
# PYTHONANYWHERE_PASSWORD, PYTHONANYWHERE_USERNAME, DATABASE_HOSTNAME, DATABASE_NAME, DATABASE_PASSWORD, DROPBOX_KEY
# and we don't want these saved to the git so I have added secret.py to .gitignore
import sys 
import os
import pdb
from secret import *
import pymysql
import sshtunnel
import platform 
import argparse 
from sqlitedict import SqliteDict
import dropbox
import multiprocessing as mp

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
               n_evals INT NOT NULL,
               PRIMARY KEY ( job_ID )
            );'''
            mycursor = mydb.cursor()
            mycursor.execute(create_db_command)
            sql = "INSERT INTO jobs (algorithm, arguments, finished, n_evals) VALUES (%s, %s, %s, %s)"
            val = (algorithm, arguments, 0, -1)
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
        sql = "DROP TABLE jobs"
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

def set_SRGP_job_finished(n_evals, job_ID):
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
        sql = "UPDATE jobs SET finished = 2, n_evals = %s  WHERE job_ID = %s"
        val = (n_evals,job_ID)
        mycursor.execute(sql, val)
        mydb.commit()
        mydb.close()            

class TransferData:
    def __init__(self, access_token):
        self.access_token = access_token

    def upload_file(self, file_from, file_to):
        """upload a file to Dropbox using API v2
        """
        dbx = dropbox.Dropbox(self.access_token)

        with open(file_from, 'rb') as f:
            dbx.files_upload(f.read(), file_to)
    
def run_all_SRGP_jobs(placeholder):
    i = 0
    dropbox_trnsfer = TransferData(DROPBOX_KEY)
    for finished in range(0,2):
        job_ID, job_arguments = get_SRGP_job(finished)        
        while job_arguments is not None:
            output_db = job_arguments[2]
            if (job_arguments[0] != pySRURGS_dir+'/experiments/SRGP.py'):
                raise Exception("SQL injection?")
            try:
                sh.python(*job_arguments, _err="error.txt")
            except:
                print(sh.cat('error.txt'))
                continue
            dropbox_trnsfer.upload_file(output_db, '/'+os.path.basename(output_db))
            with SqliteDict(output_db, autocommit=True) as results_dict:
                n_evals = results_dict['n_evals']
            set_SRGP_job_finished(n_evals, job_ID)
            job_ID, job_arguments = get_SRGP_job(finished)
            print('finished a job', i)
            i = i + 1

def find_matching_SRGP_job(train):    
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
        sql = "SELECT n_evals FROM jobs WHERE arguments CONCAT('%', ' ', %s, ' ', '%') ;"
        val = (train,)
        mycursor.execute(sql, val)
        myresult = mycursor.fetchone()
        mydb.commit()
        mydb.close()
    return myresult
            
if __name__ == '__main__':
    # Read the doc string at the top of this script.
    # Run this script in terminal with '-h' as an argument.
    parser = argparse.ArgumentParser(prog='database.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-run_SRGP", help="run the code against all the SRGP problems in the mySQL database", action="store_true")
    #parser.add_argument("run_SRURGS", help="run the code against all the SRURGS problems in the mySQL database")
    parser.add_argument("-purge_db", help="deletes all the jobs in the database", action="store_true")
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()
    if arguments.run_SRGP and arguments.purge_db:
        raise Exception("Cannot do both run SRGP jobs and purge database")
    if arguments.run_SRGP:
        pool = mp.Pool()
        pool.map(run_all_SRGP_jobs, [None]*mp.cpu_count())
    if arguments.purge_db:
        purge_db()
