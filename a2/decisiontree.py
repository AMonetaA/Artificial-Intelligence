import numpy as np
import os
from collections import Counter
from queue import PriorityQueue
import sys
from collections import defaultdict
from heapq import heappop, heappush
from matplotlib import pyplot as plt

words_dic = []

class Node:
    def __init__(self, estimate, feature, info_gain, data) -> None:
        self.id = 0
        self.estimate = estimate
        self.feature = feature
        self.info_gain = info_gain
        self.left = None
        self.right = None
        self.data = data
        self.label = -1
        self.isLeaf = True

    def print_tree(self, indent = ""):
        if self.isLeaf:
            print(f"{indent}LEAF: Prediction = {self.estimate}")
        else:
            print(f"{indent}NODE: Feature = {words_dic[self.feature-1]}, IG = {-self.info_gain}")
            # print left child
            if self.left is not None:
                print(f"{indent}  Left - {words_dic[self.feature-1]} contains:")
                self.left.print_tree(indent + "    ")
            # print right child
            if self.right is not None:
                print(f"{indent}  Right - {words_dic[self.feature-1]} does not contain:")
                self.right.print_tree(indent + "    ")
    
    # For compare purpose
    def __lt__(self, other):
        return (self.info_gain < other.info_gain)
    
    def __le__(self, other):
        return (self.info_gain <= other.info_gain)
    
   

class Document:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.words = [int]


def pointEstimate(E:list[Document]):
    N = len(E)
    if N == 0:
        return -1
    
    counter1 = len([doc for doc in E if doc.label == 1])
    counter2 = len([doc for doc in E if doc.label == 2])
    return 1 if counter1 >= counter2 else 2



entropy = lambda p: -p * np.log2(p)

def count_labels(E):
    result = {}
    for e in E:
        result

def calculate_info(E:list[Document]):
    # E has a specific feature after split
    e = [doc.label for doc in E]
    N = len(e)
    stat = Counter(e)
    if N:
        return sum([entropy(freq/N) for freq in stat.values()])
    else:
        return 1

def calculate_esplit(E1, E2 ,method:int):
    N1 = len(E1)
    N2 = len(E2)
    N = N1 + N2
    if N == 0 :
        return 1
    IE1 = calculate_info(E1)
    IE2 = calculate_info(E2)
    result = -1
    if method == 1:
        result =  1/2 * IE1 + 1/2 * IE2
    elif method == 2:
        result = N1 / N * IE1 + N2 / N * IE2
    return result


def get_words(E:list[Document]):
    # get the all the words in E
    feature = set()
    for e in E:
        for w in e.words:
            feature.add(w)
    return feature



def best_info_word(E:list[Document], words:int, method:int):
    IE = calculate_info(E)
    best_info_gain = -1
    best_word = -1
    # split on each words
    for w in get_words(E):
        # split by feature (word id)
        E_contain = set([doc for doc in E if w in doc.words])
        E_not_contain =  set([doc for doc in E if w not in doc.words])
        # find the best information gain
        esplit = calculate_esplit(E_contain, E_not_contain, method)
        info_gain = IE - esplit
        if info_gain > best_info_gain:
            best_word = w
            best_info_gain = info_gain
        # break tie
        elif info_gain == best_info_gain:
            if w < best_word:
                best_word = w
                best_info_gain = info_gain 

    return best_word, best_info_gain


def build_tree(E:list[Document], word_num:int, max_nodes, method:int):
    # root node
    w, I = best_info_word(E, word_num, method)
    start_node = Node(pointEstimate(E), w, -I, E)

    pq = [start_node]
    node_num = 0
    while node_num < max_nodes and pq:
        curr_node = heappop(pq)
        curr_node.id = node_num
        print(node_num, curr_node.feature)

        # left node contains the word
        E1 = set([doc for doc in curr_node.data if curr_node.feature in doc.words])
        w1, I1 = best_info_word(E1, word_num, method)
        pe1 = pointEstimate(E1)
        node1 = Node(pe1, w1, -I1, E1)
        heappush(pq, node1)

        # right node does not contains the word
        E2 = set([doc for doc in curr_node.data if curr_node.feature not in doc.words])
        w2, I2 = best_info_word(E2, word_num, method)
        pe2 = pointEstimate(E2)
        node2 = Node(pe2, w2, -I2, E2)
        heappush(pq, node2)

        # append the child
        curr_node.left = node1
        curr_node.right = node2
        curr_node.isLeaf = False
        node_num += 1

    return start_node

def predict(decisionTree:Node, data:Document, size):
    root = decisionTree
    while not root.isLeaf and (size == 0 or root.id < size):
        if root.feature in data.words:
            root = root.left
        else:
            root = root.right
    return root.estimate

    
def verification(decisionTree:Node, data:list[Document], size):
    correct = 0
    incorrect = 0
    # predict each document
    for doc in data:
        result = predict(decisionTree, doc, size)
        if result == doc.label:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)

def plot_accuracy(trainData, testData, decisionTree, method, size):
    suffix = 'Average'if method == 1 else 'Weighted'
    test_acc = []
    train_acc = []

    for i in range(size):
        treesize = i+1
        test_acc.append(verification(decisionTree, testData, treesize))
        train_acc.append(verification(decisionTree, trainData, treesize))

    filename = './accuracyPlot'+suffix+'.png'
    size = list(range(1,size+1))
    plt.figure()
    plt.plot(size, train_acc, label='training accuracy', color='blue')
    plt.plot(size, test_acc, label='testing accuracy', color='red')
    plt.legend(loc='lower right')
    plt.xlabel('Number of nodes in decision tree')
    plt.ylabel('Accuracy')
    plt.title(suffix + ' information gain')
    plt.savefig(filename)




def load_document(data_path, label_path):
    
    # read label
    labels = []
    with open(f'./a2/dataset/{label_path}', 'r', encoding='utf-8') as flabel:
        for line in flabel:
            val = int(line.strip())
            labels.append(val)

    # read data, store them in a dictionary
    data = {}
    with open(f'./a2/dataset/{data_path}', 'r', encoding='utf-8') as fdata:
        for line in fdata:
            id, number = map(int, line.strip().split())
            if id in data:
                data[id].append(number)
            else:
                data[id] = [number]

    # create each document using id, label and words
    # store the document in a list
    doc_list = []
    for key, value in data.items():
        if key < len(labels):
            d = Document(key, labels[key])
            d.words = value
            doc_list.append(d)

    return doc_list

def load_words(path):
    # create word directory
    with open(f'./a2/dataset/{path}', 'r', encoding='utf-8') as file:
        for line in file:
            val = line.strip()
            words_dic.append(val)
    return words_dic, len(words_dic)


def main():
    train_data = load_document('trainData.txt', 'trainLabel.txt')
    test_data = load_document('testData.txt', 'testLabel.txt')
    words, word_num = load_words('words.txt')

    # build decision tree
    # tree1 = build_tree(train_data, word_num, 100, 1)
    # plot_accuracy(train_data, test_data, tree1, 1, 100)
    # root_node.print_tree()

    tree2 = build_tree(train_data, word_num, 100, 2)
    plot_accuracy(train_data, test_data, tree2, 2, 100)



if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())
    main()