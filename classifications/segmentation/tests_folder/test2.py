#In the context of command-line arguments, a parser is a program or module that processes the arguments passed to a script or program from the command line. 
#It recognizes the structure of the command-line arguments and extracts the values of the arguments for use by the program.
#The argparse module in Python provides a built-in parser that can be used to process command-line arguments. 

import argparse

# create an argument parser object
parser = argparse.ArgumentParser()

# add an argument to the parser
parser.add_argument('--name', help='enter your name')

# parse the arguments
args = parser.parse_args()

# print the value of the name argument
print('Hello, {}!'.format(args.name))



