

class Parent:
    def __init__(self, value):
        self.value = value
        self.count = 0

    def show(self):
        self.count += 1
        print(f"Count from parent: {self.count}")


class Child(Parent):
    def __init__(self, value, child_value):
        super().__init__(value)  # Call the parent class's constructor
        self.child_value = child_value

    def show(self):
        self.count += 2
        print(f"Count from child: {self.count}")


class Sister(Parent):
    def __init__(self, value, child_value):
        super().__init__(value)
        self.child_value = child_value

    def show(self):
        self.count += 3
        print(f"Count from sister: {self.count}")


if __name__ == "__main__":
    parent = Parent(10)
    parent.show()

    child = Child(10, 20)
    child.show()

    sister = Sister(10, 20)
    sister.show()

