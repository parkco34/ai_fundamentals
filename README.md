# WBC DECISION TREE
*Predicting breast cancer status: Benign or Malignant*

## **Information Gain**
1. Entropy:
    > Average level of uncertainty, which informs us which attribute to
    split. \
    We want the lowest entropy.
    
    - Probabilities of each class
    - Binary entropy function
    - Inputs: attribute (feature)
    - Output: number (float)
    
2. Info Gain:
    > Reduction in entropy for attribute.

    - Calculate difference between entropy of parent node and weighted sum of
    child nodes.
    - Higher information gain indicates better attribute for spltting.
    Inputs: 
        X: (array-like) Attributes
        y: (array-like) Target values
        attribute: (int) Index for attribute

    Output: 
        float: information gain value

## **Build Tree**:
1. Extract unique values of the best attribute via highest information
   gain.
2. Create tree node corresponding to current split.
3. Determine remaining attributes for splitting by excluding current
   splitting attribute for further splits in the child nodes, ensuring
   each attribute is used once per path.
4. Split dataset. For each unique value of best attribute, create a subset
   of the dataset where the attribute has that value. Recursively build tree
   for each subset, creating child nodes as necessary.
5. Recursive tree building function for combining all the steps into one
   recursive function that builds the tree from root to node down to leaf
   nodes, handling stopping conditions.

[^1]: [Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
 
