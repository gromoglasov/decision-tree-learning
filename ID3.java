// ECS629/759 Assignment 2 - ID3 Skeleton Code
// Author: Simon Dixon

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.*;

class ID3 {

	/** Each node of the tree contains either the attribute number (for non-leaf
	 *  nodes) or class number (for leaf nodes) in <b>value</b>, and an array of
	 *  tree nodes in <b>children</b> containing each of the children of the
	 *  node (for non-leaf nodes).
	 *  The attribute number corresponds to the column number in the training
	 *  and test files. The children are ordered in the same order as the
	 *  Strings in strings[][]. E.g., if value == 3, then the array of
	 *  children correspond to the branches for attribute 3 (named data[0][3]):
	 *      children[0] is the branch for attribute 3 == strings[3][0]
	 *      children[1] is the branch for attribute 3 == strings[3][1]
	 *      children[2] is the branch for attribute 3 == strings[3][2]
	 *      etc.
	 *  The class number (leaf nodes) also corresponds to the order of classes
	 *  in strings[][]. For example, a leaf with value == 3 corresponds
	 *  to the class label strings[attributes-1][3].
	 **/
	class TreeNode {

		TreeNode[] children;
		int value;

		public TreeNode(TreeNode[] ch, int val) {
			value = val;
			children = ch;
		} // constructor

		public String toString() {
			return toString("");
		} // toString()
		
		String toString(String indent) {
			if (children != null) {
				String s = "";
				for (int i = 0; i < children.length; i++)
					s += indent + data[0][value] + "=" +
							strings[value][i] + "\n" +
							children[i].toString(indent + '\t');
				return s;
			} else
				return indent + "Class: " + strings[attributes-1][value] + "\n";
		} // toString(String)

	} // inner class TreeNode

	private int attributes; 	// Number of attributes (including the class)
	private int examples;		// Number of training examples
	private TreeNode decisionTree;	// Tree learnt in training, used for classifying
	private String[][] data;	// Training data indexed by example, attribute
	private String[][] strings; // Unique strings for each attribute
	private int[] stringCount;  // Number of unique strings for each attribute

	public ID3() {
		attributes = 0;
		examples = 0;
		decisionTree = null;
		data = null;
		strings = null;
		stringCount = null;
	} // constructor
	
	public void printTree() {
		if (decisionTree == null)
			error("Attempted to print null Tree");
		else
			System.out.println(decisionTree);
	} // printTree()

	/** Print error message and exit. **/
	static void error(String msg) {
		System.err.println("Error: " + msg);
		System.exit(1);
	} // error()

	static final double LOG2 = Math.log(2.0);
	
	static double xlogx(double x) {
		return x == 0? 0: x * Math.log(x) / LOG2;
	} // xlogx()

	/** Execute the decision tree on the given examples in testData, and print
	 *  the resulting class names, one to a line, for each example in testData.
	 **/
	public void classify(String[][] testData) {
		if (decisionTree == null) error("Please run training phase before classification");
		// classifies each row in the dataset and prints it
		for(int i = 1; i < testData.length; i++){
			int result = classify(testData[i], decisionTree);
			System.out.println(strings[attributes-1][result]);
		}
	} // classify()

	private int classify(String[] testData, TreeNode tree){
		// if leaf node was reached
		if(tree.children == null) return tree.value;

		// get string of the attribute for the given tree node
		String valueString = testData[tree.value];

		// recursively follows the tree until the end of the tree is reached
		for(int i = 0; i < stringCount[tree.value]; i++){
			if(valueString.equals(strings[tree.value][i])) return classify(testData, tree.children[i]);
		}

		// training data does not have the given string
		return mostCommonNodeValue(tree);
	}

	// find them most common label from the remaning nodes of the tree
	public int mostCommonNodeValue(TreeNode tree) {
		int[] values = mostCommonRec(tree);
		int maxValue = -1;
		int maxCounter = -1;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > maxCounter) {
				maxValue = i;
				maxCounter = values[i];
			}
		}
		return maxValue;
	}

	// recursive goes through the tree to find the most common label
	public int[] mostCommonRec(TreeNode tree) {
		int[] mostCommonValues = new int[stringCount[attributes-1]];
		if (tree.children == null) {
			mostCommonValues[tree.value]++;
			return mostCommonValues;
		}
		int[] values;
		for (int i = 0; i < tree.children.length; i++) {
			values = mostCommonRec(tree.children[i]);
			for (int j = 0; j < values.length; j++) mostCommonValues[j] += values[j];
		}

		return mostCommonValues;
	}

	public void train(String[][] trainingData) {
		indexStrings(trainingData);
		decisionTree = buildTree(data, trainingData, new ArrayList<Integer>());

	} // train()

	public TreeNode buildTree(String[][] workingData, String[][] parentData, ArrayList<Integer> droppedAttributes) {
		// exit condition if there are no more examples
		if (workingData.length == 1) return new TreeNode(null, mostCommonValue(parentData));

		// exit condition if there are no more attributes
		if (droppedAttributes.size() == attributes - 1) return new TreeNode(null, mostCommonValue(workingData));

		// exit condition if all of the remaning training examples have the same target values
		if (!canPartition(workingData)) return new TreeNode(null, mostCommonValue(workingData));

        // calculate the best attribute to partition the data by
		int bestAttribute = getBestAttribute(workingData, droppedAttributes);

		// get the new data for each child node after partitioning
		ArrayList<String[][]> childrenData = getChildrenData(workingData, bestAttribute);
		TreeNode[] children = new TreeNode[childrenData.size()];
		ArrayList<Integer> updatedDropAttributes = new ArrayList<Integer>();

		// add the best attribute to the list of the attributes used in previous nodes
		for(Integer attribute : droppedAttributes) updatedDropAttributes.add(attribute);
		updatedDropAttributes.add(bestAttribute);

		// recursive call to each of the children
		for (int i = 0; i < children.length; i++) {
			children[i] = buildTree(childrenData.get(i), workingData, updatedDropAttributes);
		}
		// returns new node with its children and the attribute used to split the data
		return new TreeNode(children, bestAttribute);
	}

	// splits the data into groups by the provided attribute number of groups is equal to the number of unique value
	// for the given attribute
	public ArrayList<String[][]> getChildrenData(String[][] remainingData, int columnToRemove) {
		ArrayList<String> splits = getSplits(remainingData, columnToRemove);
		ArrayList<String[][]> childrenData = new ArrayList<String[][]>();
		String[][] childData;
		ArrayList<String[]> cleanedChildData = new ArrayList<String[]>();
		String[] rowData = new String[remainingData[0].length - 1];
		int counter = 0;
		for (String split : splits) {
			childData = partitionData(split, remainingData, columnToRemove);
			cleanedChildData = new ArrayList<String[]>();
			for (int i = 0; i < childData.length; i++) {
				cleanedChildData.add(childData[i]);
			}
			childrenData.add(cleanedChildData.toArray(new String[cleanedChildData.size()][cleanedChildData.get(0).length]));
		}
		return childrenData;
	}

	// return the most common target value from a give dataset
	public int mostCommonValue(String[][] workingData) {
		int[] possibleValues = new int[stringCount[stringCount.length-1]];

		ArrayList<String> possibleStrings = new ArrayList<String>();
		for (int i = 0; i < possibleValues.length; i++) {
			possibleValues[i] = i;
			possibleStrings.add(strings[strings.length-1][i]);
		}
		int[] valueCounter = new int[possibleValues.length];
		for (int i = 1; i < workingData.length; i++) valueCounter[possibleStrings.indexOf(workingData[i][workingData[i].length-1])]++;

		int maxValue = -1;
		int maxCounter = -1;

		for (int i = 0; i < valueCounter.length; i++) {
			if (valueCounter[i] > maxCounter) {
				maxValue = i;
				maxCounter = valueCounter[i];
			}
		}

		return maxValue;
	}

	// checks if all of the data has the same target value
	public boolean canPartition(String[][] remainingData)
	{
		String firstResponse = remainingData[1][remainingData[1].length-1];
		String nextResponse;

		for (int i = 1; i < remainingData.length; i++) {
			nextResponse = remainingData[i][remainingData[i].length-1];
			if(!remainingData[i][remainingData[i].length-1].equals(firstResponse)) {
				return true;
			}
		}
		return false;
	}


	public String[][] partitionData(String split, String[][] remainingData, int attribute) {
		ArrayList<String[]> partitionedDataList = new ArrayList<String[]>();
		partitionedDataList.add(remainingData[0]);
		for (int i = 1; i < remainingData.length; i++) if (remainingData[i][attribute].equals(split)) partitionedDataList.add(remainingData[i]);
		String[][] partitionedData = new String[partitionedDataList.size()][remainingData[1].length];
		for (int i = 0; i < partitionedDataList.size(); i++) partitionedData[i] = partitionedDataList.get(i);
		return partitionedData;
	}

	// calculates the impurity of the given dataset => returns the probability (1 - correct random classification)
	public double calculateImpurity(String[][] remainingData) {
		int numberOfExamples = remainingData.length - 1;
		ArrayList<String> labels = new ArrayList<String>();
		ArrayList<Integer> count = new ArrayList<Integer>();
		for (int i = 1; i < remainingData.length; i++) {
			if (!labels.contains(remainingData[i][remainingData[i].length - 1])) {
				labels.add(remainingData[i][remainingData[i].length - 1]);
				count.add(1);
			} else {
				count.set(labels.lastIndexOf(remainingData[i][remainingData[i].length - 1]), count.get(labels.lastIndexOf(remainingData[i][remainingData[i].length - 1])) + 1);
			}
		}
		double correctRandomClassification = 0;
		double randomClass;
		for (int i = 0; i < count.size(); i++) {
			randomClass = count.get(i).doubleValue()/numberOfExamples;
			randomClass *= randomClass;
			correctRandomClassification += randomClass;
		}
		return 1 - correctRandomClassification;
	}

	// calculate average impurity => average of potential children's impurities
	public double calculateAverageImpurity(ArrayList<String> splits, String[][] remainingData, int attributeNumber) {
		String[][] partitionedData;
		int numberOfExamples = remainingData.length - 1;
		double[] weights = new double[splits.size()];
		double[] impurities = new double[splits.size()];
		for (int i = 0; i < splits.size(); i++) {
			partitionedData = partitionData(splits.get(i), remainingData, attributeNumber);
			weights[i] = (double)(partitionedData.length - 1) / numberOfExamples;
			impurities[i] = calculateImpurity(partitionedData);
		}
		double averageImpurity = 0;
		for (int i = 0; i < weights.length; i++) averageImpurity += (weights[i] * impurities[i]);
		return averageImpurity;
	}

	// calculates information gain for using a given attribute to split the data
	public double calculateInformationGain(int attributeNumber, String[][] remainingData) {
		// initial impurity is calculated
		double initialImpurity = calculateImpurity(remainingData);

		ArrayList<String> splits = getSplits(remainingData, attributeNumber);

		double impurityAfterSplit = calculateAverageImpurity(splits, remainingData, attributeNumber);

		return initialImpurity - impurityAfterSplit;
	}

	// returns the possible values for the given data at a given attribute
	public ArrayList<String> getSplits(String[][] remainingData, int attributeNumber) {
		String[] splits = strings[attributeNumber];
		ArrayList<String> splitsList = new ArrayList<String>();
		for (int i = 0; i < splits.length; i++){
			if(splits[i] == null) break;
			splitsList.add(splits[i]);
		}
		return splitsList;
	}

	// returns the best attribute to split the data by
	public int getBestAttribute(String[][] remainingData, ArrayList<Integer> droppedAttributes) {
		double bestInformationGain = 0;
		double informationGain = 0;
		int bestAttribute = 0;
		for (int i = 0; i < attributes - 1; i++) {
			if (droppedAttributes.indexOf(i) == -1) {
				informationGain = calculateInformationGain(i, remainingData);
				if (informationGain > bestInformationGain) {
//					System.out.println("New information Gain: " + informationGain);
					bestInformationGain = informationGain;
					bestAttribute = i;
				} else {
//					System.out.println("information Gain: " + informationGain + "is lower than " + bestInformationGain);
				}
			} else {
//				System.out.println("The attribute has already been removed");
			}

		}
		return bestAttribute;
	}

	/** Given a 2-dimensional array containing the training data, numbers each
	 *  unique value that each attribute has, and stores these Strings in
	 *  instance variables; for example, for attribute 2, its first value
	 *  would be stored in strings[2][0], its second value in strings[2][1],
	 *  and so on; and the number of different values in stringCount[2].
	 **/
	void indexStrings(String[][] inputData) {
		data = inputData;
		examples = data.length;
		attributes = data[0].length;
		stringCount = new int[attributes];
		strings = new String[attributes][examples];// might not need all columns
		int index = 0;
		for (int attr = 0; attr < attributes; attr++) {
			stringCount[attr] = 0;
			for (int ex = 1; ex < examples; ex++) {
				for (index = 0; index < stringCount[attr]; index++)
					if (data[ex][attr].equals(strings[attr][index]))
						break;	// we've seen this String before
				if (index == stringCount[attr])		// if new String found
					strings[attr][stringCount[attr]++] = data[ex][attr];
			} // for each example
		} // for each attribute
	} // indexStrings()

	/** For debugging: prints the list of attribute values for each attribute
	 *  and their index values.
	 **/
	void printStrings() {
		for (int attr = 0; attr < attributes; attr++)
			for (int index = 0; index < stringCount[attr]; index++)
				System.out.println(data[0][attr] + " value " + index +
									" = " + strings[attr][index]);
	} // printStrings()
		
	/** Reads a text file containing a fixed number of comma-separated values
	 *  on each line, and returns a two dimensional array of these values,
	 *  indexed by line number and position in line.
	 **/
	static String[][] parseCSV(String fileName)
								throws FileNotFoundException, IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String s = br.readLine();
		int fields = 1;
		int index = 0;
		while ((index = s.indexOf(',', index) + 1) > 0)
			fields++;
		int lines = 1;
		while (br.readLine() != null)
			lines++;
		br.close();
		String[][] data = new String[lines][fields];
		Scanner sc = new Scanner(new File(fileName));
		sc.useDelimiter("[,\n]");
		for (int l = 0; l < lines; l++)
			for (int f = 0; f < fields; f++)
				if (sc.hasNext())
					data[l][f] = sc.next();
				else
					error("Scan error in " + fileName + " at " + l + ":" + f);
		sc.close();
		return data;
	} // parseCSV()

	public static void main(String[] args) throws FileNotFoundException,
												  IOException {
		if (args.length != 2)
			error("Expected 2 arguments: file names of training and test data");
		String[][] trainingData = parseCSV(args[0]);
		String[][] testData = parseCSV(args[1]);
		ID3 classifier = new ID3();
		classifier.train(trainingData);
		classifier.printTree();
		classifier.classify(testData);
	} // main()

} // class ID3
