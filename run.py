import os
# to do experiment or with no of combination of hyperparameter
#its like custom grid search 


n_estimators =[110,100,150,200,210]
max_depth =[20,25,15,10,5]


for n in n_estimators:
    for m in max_depth:
        os.system(f"python basic_ml_model.py -n{n} -m{m}")
        #os.system run python basic_ml_model.py command in cmd terminal with n ,m values passed here 