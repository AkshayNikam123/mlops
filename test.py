import argparse #bult in library in python
 #to pass argument from command prompt this method is used
 # for experimenting you can use this by passing different arg i.e to function in command line


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    #specifying argument
    args.add_argument("--name", "-n", default="akshay", type=str)
    args.add_argument("--age", "-a", default=27.0, type=float)
    parse_args=args.parse_args()
    
    print(parse_args.name,parse_args.age)

    # - to call by short  
    # -- to get full not short 

    #command line 
    #python test.py
    #output akshay 27

    #python test.py -n "amit"
    #output amit 27
    #even it is set with default you can change like this 

    #python test.py --name "amit"
    #output amit 27

    #python test.py -n "amit" -a 56
    #output amit 56 