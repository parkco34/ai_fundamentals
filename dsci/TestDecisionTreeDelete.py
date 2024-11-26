#!/usr/bin/env python
import numpy as np
import pandas as pd
from decision_tree import DecisionTree  # Assuming your code is in decision_tree.py
import unittest

class TestDecisionTreeDelete(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test method."""
        # Load the exam results data
        self.df = pd.read_csv('exam_results.csv')

        # Convert categorical variables to numeric
        self.X = pd.get_dummies(self.df.drop(['Resp srl no', 'Exam Result'], axis=1))
        self.y = (self.df['Exam Result'] == 'Pass').astype(int)

        # Create simple test arrays for basic function testing
        self.simple_y = np.array([0, 0, 1, 1, 1])
        self.simple_X = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1]
        ])

        # Initialize decision tree
        self.tree = DecisionTree(max_depth=3, min_num_samples=2)

    def test_get_probabilities(self):
        """Test the get_probabilities method."""
        # Test with simple binary array
        probs = self.tree.get_probabilities(self.simple_y)
        expected_probs = np.array([0.4, 0.6])  # [2/5, 3/5]

        np.testing.assert_array_almost_equal(probs, expected_probs)
        self.assertEqual(len(probs), 2)  # Should have probabilities for both classes

    def test_parent_entropy(self):
        """Test the parent_entropy method."""
        # Test with simple binary array
        entropy = self.tree.parent_entropy(self.simple_y)
        # Expected entropy calculation:
        # -(0.4 * log2(0.4) + 0.6 * log2(0.6)) ≈ 0.971
        expected_entropy = 0.971

        self.assertAlmostEqual(entropy, expected_entropy, places=3)

        # Test with pure node (all same class)
        pure_y = np.array([1, 1, 1])
        pure_entropy = self.tree.parent_entropy(pure_y)
        self.assertEqual(pure_entropy, 0.0)

    def test_child_entropy(self):
        """Test the child_entropy method."""
        # Test with simple dataset
        child_ent = self.tree.child_entropy(self.simple_X, self.simple_y, attribute=0)

        # Calculate expected entropy manually:
        # P(X=1) = 0.4, P(X=0) = 0.6
        # For X=1: P(Y=0|X=1) = 1.0, P(Y=1|X=1) = 0.0
        # For X=0: P(Y=0|X=0) = 0.0, P(Y=1|X=0) = 1.0
        expected_ent = 0.0  # Perfect split should give 0 entropy

        self.assertAlmostEqual(child_ent, expected_ent, places=3)

    def test_parent_gini(self):
        """Test the parent_gini method."""
        # Test with simple binary array
        gini = self.tree.parent_gini(self.simple_y)
        # Expected gini = 1 - (0.4^2 + 0.6^2) = 0.48
        expected_gini = 0.48

        self.assertAlmostEqual(gini, expected_gini, places=3)

        # Test with pure node
        pure_gini = self.tree.parent_gini(np.array([1, 1, 1]))
        self.assertEqual(pure_gini, 0.0)

    def test_child_gini(self):
        """Test the child_gini method."""
        gini = self.tree.child_gini(self.simple_X, self.simple_y, attribute=0)
        # Perfect split should give 0 gini impurity
        expected_gini = 0.0

        self.assertAlmostEqual(gini, expected_gini, places=3)

    def test_best_split(self):
        """Test the best_split method."""
        # Test with gini criterion
        best_feat_gini = self.tree.best_split(self.simple_X, self.simple_y, method="gini")
        self.assertEqual(best_feat_gini, 0)  # First feature should give perfect split

        # Test with entropy criterion
        best_feat_entropy = self.tree.best_split(self.simple_X, self.simple_y, method="entropy")
        self.assertEqual(best_feat_entropy, 0)  # Should also choose first feature

    def test_plurality_value(self):
        """Test the plurality_value method."""
        # Test with simple majority
        majority_class = self.tree.plurality_value(self.simple_y)
        self.assertEqual(majority_class, 1)

        # Test with tie (should break randomly but consistently with same random_state)
        tie_y = np.array([0, 0, 1, 1])
        result1 = self.tree.plurality_value(tie_y, random_state=42)
        result2 = self.tree.plurality_value(tie_y, random_state=42)
        self.assertEqual(result1, result2)

    def test_learn_decision_tree(self):
        """Test the learn_decision_tree method."""
        # Test with simple dataset
        tree = self.tree.learn_decision_tree(self.simple_X, self.simple_y)

        # Check basic tree structure
        self.assertIn('feature', tree)
        self.assertIn('branches', tree)

        # Test with actual exam dataset
        X_exam = self.X.to_numpy()
        tree_exam = self.tree.learn_decision_tree(X_exam, self.y.to_numpy())

        # Basic structure checks
        self.assertIsInstance(tree_exam, dict)
        self.assertTrue('feature' in tree_exam or 'class' in tree_exam)

if __name__ == '__main__':
    unittest.main()

