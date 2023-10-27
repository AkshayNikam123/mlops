import mlflow
 #just for understanding mlflow working we created this file
 #ml flow is used track experiment 
 #object like config artifact model

def calculate_sum(x,y):
    return x*y




if __name__=="__main__":
  with mlflow.start_run():
    #start the server
    x,y=100,200
    z=calculate_sum(x,y)
    print(f"sum is:{z}")
    #track log
    mlflow.log_param("x",x)
    mlflow.log_param("y",y)
    mlflow.log_metric("z",z)