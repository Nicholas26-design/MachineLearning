# Classes allow you to create your own data type. The class defines the data type and what's in it.
# Creating something based off the class is an object. An object is an instance of a class.
class Student:
    def __init__(self, name, age, major, is_repeat_student):
        # def __init__(self, name, age, major, is_repeat_student): These are parameters
        self.name = name
        self.age = age
        self.major = major
        self.is_repeat_student = is_repeat_student
# passed in values need to be assigned to the attribute of the object. That is why self.name = name.
# Object.attribute = passed in value
    def is_postgrad_student(self):
        if self.age >= 28:
            return "Yes"
        else:
            return "No"
# The above is an example of a function written within the class of Student. Notice the indentation on the IDE.
Student1 = Student("Nick", 27, "Finance", False)


print(Student1.name)
print(Student1.is_postgrad_student())




