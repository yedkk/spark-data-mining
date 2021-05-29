"""
Source: EurekaTrees (https://github.com/ChuckWoodraska/EurekaTrees)
"""

import os
import jinja2
import json

class Node(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None


class Tree(object):
    def __init__(self):
        self.root = None
        self.max_depth = None
        self.max_breadth = None

    def create_tree(self, tree, column_names):
        st = len('Predict: ')
        else_check = 0
        node = None
        for index, line in enumerate(tree['Contents']):
            if line.startswith('If'):
                data_str = line[line.find('(') + 1:line.find(')')].replace(' ', '_', 1)
                if len(column_names):
                    data_str = column_names[data_str.split('_')[1].split(' ')[0]] + ' ' + ' '.join(data_str.split('_')[1].split(' ')[1:])
                if not node:
                    node = self.root = Node(data=data_str)
                elif else_check:
                    else_check = 0
                    while node.right:
                        node = node.parent
                    node.right = Node(data=data_str)
                    node.right.parent = node
                    node = node.right
                else:
                    node.left = Node(data=data_str)
                    node.left.parent = node
                    node = node.left
            elif line.startswith('Else'):
                else_check = 1
            elif line.startswith('Predict'):
                if not node:
                    node = self.root = Node(data=line[st:])
                elif else_check:
                    else_check = 0
                    while node.right:
                        node = node.parent
                    node.right = Node(data=line[st:])
                    node.right.parent = node
                    node = node.parent
                else:
                    node.left = Node(data=line[st:])
                    node.left.parent = node
        self.max_depth = self.get_max_depth(self.root) - 1
        self.max_breadth = self.get_max_breadth(self.max_depth)

    def print_inorder(self, node):
        if node is not None:
            self.print_inorder(node.left)
            print(node.data)
            self.print_inorder(node.right)

    def preorder(self, node, node_list=None):
        if node_list is None:
            node_list = []
        if node is not None:
            node_list.append(node)
            self.preorder(node.left, node_list)
            self.preorder(node.right, node_list)
        return node_list

    def get_js_struct(self, node, node_dict=None):
        if node_dict is None:
            node_dict = {'name': node.data, 'children': []}
        if node is not None:
            if node.left:
                new_node_dict_left = {'name': node.left.data, 'type': 'left', 'is_prediction': False, 'children': []}
                node_dict['children'].append(self.get_js_struct(node.left, new_node_dict_left))
            if node.right:
                new_node_dict_right = {'name': node.right.data, 'type': 'right', 'is_prediction': False, 'children': []}
                node_dict['children'].append(self.get_js_struct(node.right, new_node_dict_right))
            else:
                node_dict['is_prediction'] = True
                node_dict['na'] = node.data
                node_dict['value'] = node.data
            
            if node.parent is None:
                node_dict['type'] = 'root'
            
            
            if node.left or node.right:
                node_dict['op'] = node.data.split(' ')[-2]
                node_dict['value'] = node.data.split(' ')[-1]
                node_dict['na'] = node.data[:-len(node_dict['op']) - len(node_dict['value']) - 1]
            
        return node_dict

    def print_preorder(self, node):
        if node is not None:
            print(node.data)
            self.print_preorder(node.left)
            self.print_preorder(node.right)

    def print_postorder(self, node):
        if node is not None:
            self.print_postorder(node.left)
            self.print_postorder(node.right)
            print(node.data)

    def get_max_depth(self, node):
        if node is None:
            return 0
        else:
            left_depth = self.get_max_depth(node.left)
            right_depth = self.get_max_depth(node.right)
            if left_depth > right_depth:
                return left_depth + 1
            else:
                return right_depth + 1

    def get_max_breadth(self, max_depth=None):
        if max_depth is None:
            max_depth = self.get_max_depth(self.root)
        return 2 ** max_depth

def make_tree_viz(trees, output_path):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader([os.path.dirname(os.path.abspath(__file__))]))
    tree_template = env.get_template("tree_template.jinja2")
    for index, tree in enumerate(trees):
        result = tree_template.render(tree=json.dumps(tree['tree']),
                                      max_depth=tree['max_depth'] * 120 if tree['max_depth'] else 120,
                                      max_breadth=tree['max_depth'] * 750 if tree['max_depth'] else 750)
        with open(output_path, 'w') as tree_html:
            tree_html.write(result)

def plot_trees(trees, column = {}, output_path='tree.html'):
    trees = separate_trees(trees)
    tree_list = []
    for index, tree in enumerate(trees):
        tree_obj = Tree()
        tree_obj.create_tree(tree, column)
        js_struct = tree_obj.get_js_struct(tree_obj.root)
        node_dict = {'tree': [js_struct], 'max_depth': tree_obj.max_depth, 'max_breadth': tree_obj.max_breadth}
        tree_list.append(node_dict)
    make_tree_viz(tree_list, output_path)
    
def separate_trees(tree_file):
    tree = ''
    tree_contents = []
    tree_list = []
    for line in tree_file:
        line = line.strip().rstrip()
        if line.find('Tree') != -1:
            if tree:
                tree_list.append({'Tree': tree, 'Contents': tree_contents})
            tree = line
            tree_contents = []
        else:
            tree_contents.append(line)
    tree_list.append({'Tree': tree, 'Contents': tree_contents})
    return tree_list