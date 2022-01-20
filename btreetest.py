class Node:
    def __init__(self, ndx, val):
        self.left = None
        self.right = None

        # Assume data has two classes, data.idx and data.val
        # But could also just have the class which is data and
        self.ndx = ndx
        self.val = val

    def insert(self, ndx, val):
        if self.ndx:
            if ndx < self.ndx:
                if self.left is None:
                    self.left = Node(ndx, val)
                else:
                    self.left.insert(ndx, val)

            elif ndx > self.ndx:
                if self.right is None:
                    self.right = Node(ndx, val)
                else:
                    self.right.insert(ndx, val)
            else:
                assert self.ndx == ndx
                self.val = val

    def findval(self, ndx):
        if ndx < self.ndx:
            if self.left:
                self.left.findval(ndx)
            else:
                print(f"Can't find val for index {ndx}.")
        elif ndx > self.ndx:
            if self.right:
                self.right.findval(ndx)
            else:
                print(f"Can't find val for index {ndx}.")
        else:
            assert ndx == self.ndx
            print(f"Data is {str(self.val)}")

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()

        print(f"  {self.ndx:4}: {str(self.val)}")

        if self.right:
            self.right.PrintTree()
