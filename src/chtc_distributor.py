import os
import sys

def process_jobs(file_dir):
    """
    Reads list of python jobs from a text file, where each job is on a new 
    line. It then runs the python job according the environemnt variable
    $process.
    """
    python_jobs = None
    try:
        f = open(file_dir, 'r')
        python_jobs = f.readlines()
        python_jobs = [job.strip() for job in python_jobs] 
    except IOError:
        print('Error: Cannot open python job file ' + file_dir)
        return  
    
    process_id = os.environ.get('process')
    if process_id == None:
        print('Error: No environemnt variable $process exists.')
        return    
    if process_id >= len(python_jobs) or process_id < 0:
        print('Error: Process id value invalid range: ' + str(process_id))
        return
    
    os.system(python_jobs[process_id])


if __name__ == "__main__":
    job_file_dir = "python_job_scripts.txt"
    if len(sys.argv) > 1:
        job_file_dir = sys.argv[1]   
    
    process_jobs(job_file_dir)