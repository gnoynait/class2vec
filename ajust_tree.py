class TreeNode(object):
    def __init__(self, name):
        self.name = name
        self.children = []
    def add(self, path, depth):
        if depth >= len(path):
            return
        name = path[depth]
        pos = len(self.children)
        for i, child in enumerate(self.children):
            if child.name == name:
                pos = i
        if pos == len(self.children):
            self.children.append(TreeNode(name))
        self.children[pos].add(path, depth + 1)
    def ajust(self):
        if len(self.children) == 0:
            return
        elif len(self.children) == 1:
            if self.name == '':
                self.name = self.children[0].name
            else:
                self.name = self.name +'.' +  self.children[0].name
            self.children = self.children[0].children
            self.ajust()
        elif len(self.children) == 2:
            self.children[0].ajust()
            self.children[1].ajust()
        elif len(self.children) == 3:
            right = self.children[1:]
            self.children = [self.children[0], TreeNode('<dummy>')]
            self.children[0].ajust()
            self.children[1].children = right
            self.children[1].ajust()
        else:
            mid = len(self.children) / 2
            left = self.children[:mid]
            right = self.children[mid:]
            self.children = [TreeNode('<dummy>'), TreeNode('<dummy>')]
            self.children[0].children = left
            self.children[1].children = right
            self.children[0].ajust()
            self.children[1].ajust()
    def output_map(self, path = '', prefix = ''):
        if len(self.children) == 0:
            print path + '.' + self.name, prefix
            return
        if path == '':
            path = self.name
        else:
            path = path + '.' + self.name
        self.children[0].output_map(path, prefix + '0')
        self.children[1].output_map(path, prefix + '1')
        

def buildTree(tree_file):
    root = TreeNode('')
    for line in tree_file:
        line = line.strip()
        if line == '':
            continue
        row = line.split('.')
        root.add(row, 0)
    return root
import sys
root = buildTree(sys.stdin)
root.ajust()
root.output_map()
