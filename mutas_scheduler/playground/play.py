
class Person:
    def __init__(self):
        self.name = "Jack"
        self.age = 20

person = Person()

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def foo(data):
    for k, v in data.items():
        setattr(person, k, v)


if __name__ == '__main__':
    prn_obj(person)
    data = {'name': 'Mike', 'age': int("100")}
    foo(data)
    prn_obj(person)
    print("over")