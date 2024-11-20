#!/usr/bin/env python
def predict(tree, test, indent=""):
    """
    Predictions with debugging output
    --------------------------------
    INPUT:
        tree: (dict) Decision Tree model
        test: (dict) Test data
        indent: (str) For pretty printing the debug output
    """
    print(f"{indent}Current tree node:", tree)

    # If we've reached a leaf node (non-dictionary)
    if not isinstance(tree, dict):
        print(f"{indent}Reached leaf node with prediction:", tree)
        return tree

    # If we've reached a node with only Pass/Fail counts
    if set(tree.keys()) <= {"Pass", "Fail"}:
        prediction = "Pass" if tree["Pass"] > tree["Fail"] else "Fail"
        print(f"{indent}Reached leaf node with counts - Pass: {tree.get('Pass', 0)}, Fail: {tree.get('Fail', 0)}")
        print(f"{indent}Predicting:", prediction)
        return prediction

    for attribute, subtree in tree.items():
        print(f"{indent}Testing attribute: {attribute}")
        next_attribute = test[attribute]
        print(f"{indent}Test case has value: {next_attribute}")

        if next_attribute in subtree:
            print(f"{indent}Following path: {attribute} = {next_attribute}")
            return predict(subtree[next_attribute], test, indent + "  ")
        else:
            print(f"{indent}WARNING: Value {next_attribute} not found in training data for {attribute}")
            # Return majority class or handle unseen values
            majority = max(("Pass", "Fail"),
                         key=lambda x: sum(1 for _, v in subtree.items()
                                         if isinstance(v, dict) and
                                         v.get(x, 0) > v.get("Fail" if x == "Pass" else "Pass", 0)))
            print(f"{indent}Using majority class: {majority}")
            return majority

    # Should never reach here if tree is properly constructed
    print(f"{indent}ERROR: No valid attribute found in tree node")
    return None
