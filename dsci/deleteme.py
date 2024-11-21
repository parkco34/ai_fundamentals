#!/usr/bin/env python
from graphviz import Digraph

def visualize_tree(tree, parent_name=None, edge_label=None, graph=None):
    if graph is None:
        graph = Digraph(format='png')
        graph.attr('node', shape='rectangle')

    if not isinstance(tree, dict):
        # Leaf node
        node_name = f"Leaf_{id(tree)}"
        graph.node(node_name, label=f"Predict: {tree}")
        if parent_name:
            graph.edge(parent_name, node_name, label=edge_label)
    else:
        for attribute, branches in tree.items():
            node_name = f"Node_{id(attribute)}"
            graph.node(node_name, label=attribute)
            if parent_name:
                graph.edge(parent_name, node_name, label=edge_label)
            for branch_value, subtree in branches.items():
                visualize_tree(subtree, parent_name=node_name, edge_label=str(branch_value), graph=graph)
    return graph

# Build and clean the tree
our_tree = grow_tree(X, y)
cleaned_tree = clean_tree(our_tree)

# Visualize
tree_graph = visualize_tree(cleaned_tree)
tree_graph.render('decision_tree')


