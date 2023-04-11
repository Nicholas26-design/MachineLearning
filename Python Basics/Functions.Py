# The point of the function is going to be to say hi to the user.
# The keyword def begins a function
# This will also have common use cases, like directory changes.

import os # Directory Module
cwd = os.getcwd() # gets current wd
print(cwd) # shows me cwd

# See below to change the current working directory to specified path
os.chdir('c:\\gfg_dir')


# Start
# All code after the def is going to be in the function. That's seen by indentation
def say_hi(name):
    print("Hello from a function " + name)
print("top")
# In order to actually execute the function, you enter the function without indentation. So at point zero basically.
# This acts as a call.
say_hi("Nick")
say_hi("Sam")



# The point of the function is going to be to cube a number.
# We are using the return statement here. This returns info from a function
# End


# Start
# All code after the def is going to be in the function. That's seen by indentation
def cube(num):
    return num*num*num
# Return breaks you out of the code. It makes the function return a value, like the sum of something
result = cube(4)
print(result)

