






def splitTask(tasks, workers): ##allocate tasks to workers
    taskList = []
    num_tasks = len(tasks)
    for i in range(workers-1):
        taskList.append(range(i*(num_tasks/4),(i+1)*(num_tasks/4)))
    taskList.append(range((workers-1)*(num_tasks/4),num_tasks))  
    return taskList