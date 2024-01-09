# 定义了一个名为Person的类，它具有两个方法：__call__和hello。
class Person:
    # __call__方法是一个特殊方法，在类的实例被调用时会被触发。
    # 在这个例子中，__call__方法接受一个参数name，并打印出"__call__Hello"加上该参数的值。
    def __call__(self, name):
        print("__call__"+"Hello"+name)

    # hello方法是一个普通的实例方法，接受一个参数name，并打印出"hello"加上该参数的值。
    def hello(self,name):
        print("hello"+name)

# __call__方法使得类的实例可以像函数一样被调用，而普通的实例方法需要通过对象名和方法名的方式进行调用。

# 创建了一个名为person的Person类的实例。
person=Person()

# 通过将实例名后面加上括号的方式调用了person对象
# 实际上是调用了Person类的__call__方法，并传递了参数"zn"
person("zn")

# 通过调用person对象的hello方法，并传递参数"list"，会打印出"hellolist"。
person.hello("list")