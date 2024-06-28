from celery import Celery
import subprocess

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

celery_app.conf.update(
    task_soft_time_limit=300,  # Soft time limit (seconds)
    task_time_limit=320,       # Hard time limit (seconds)
)

@celery_app.task
def run_llama(intent, prompt, actions):
    command = ["python3", "/home/orlando/action-scripts/llama.py", intent, prompt, actions]
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    response = output.stdout.split('[/INST]')[1][2:]
    return response
