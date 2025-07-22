#include <bits/stdc++.h>
using namespace std;
#define el "\n"
#define ll long long
#define ull unsigned long long
#define se second
#define fi first
#define Faster cin.tie(0); cout.tie(0); ios_base::sync_with_stdio(0);

vector<string> split(const string& s, char delimiter) 
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) 
    {
        tokens.push_back(token);
    }
    return tokens;
}

void readData(const string& filePath, vector<vector<double>>& features, vector<int>& labels, bool hasLabels) 
{
    ifstream file(filePath);
    string line;
    while (getline(file, line)) 
    {
        vector<string> tokens = split(line, ',');
        vector<double> featureRow;
        if (hasLabels) 
        {
            if (tokens[0] == "L") labels.push_back(0);
            else if (tokens[0] == "R") labels.push_back(1);
            else if (tokens[0] == "B") labels.push_back(2);
            for (size_t i = 1; i < tokens.size(); ++i) 
            {
                featureRow.push_back(stod(tokens[i]));
            }
        } 
        else 
        {
            for (const auto& token : tokens) 
            {
                featureRow.push_back(stod(token));
            }
        }
        features.push_back(featureRow);
    }
    file.close();
}

double giniImpurity(const vector<int>& labels) 
{
    if (labels.empty()) return 0.0;
    vector<int> counts(3, 0);
    for (int label : labels) 
    {
        counts[label]++;
    }
    double gini = 1.0;
    for (int count : counts) 
    {
        double p = 1.0 * count / labels.size();
        gini -= p * p;
    }
    return gini;
}

struct TreeNode 
{
    int featureIndex;
    double threshold;
    int label;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : featureIndex(-1), threshold(0), label(-1), left(nullptr), right(nullptr) {}
};

TreeNode* buildTree(const vector<vector<double>>& features, const vector<int>& labels, int depth, int maxDepth, int minSamplesSplit, int minSamplesLeaf) 
{
    if (depth >= maxDepth || features.size() < minSamplesSplit) 
    {
        vector<int> counts(3, 0);
        for (int label : labels) 
        {
            counts[label]++;
        }
        TreeNode* node = new TreeNode();
        node->label = max_element(counts.begin(), counts.end()) - counts.begin();
        return node;
    }
    int bestFeature = -1;
    double bestThreshold = 0;
    double bestGini = 1.0;
    for (size_t i = 0; i < features[0].size(); ++i) 
    {
        for (const auto& row : features) 
        {
            double threshold = row[i];
            vector<int> leftLabels, rightLabels;
            for (size_t j = 0; j < features.size(); ++j) 
            {
                if (features[j][i] <= threshold) {
                    leftLabels.push_back(labels[j]);
                } else {
                    rightLabels.push_back(labels[j]);
                }
            }
            if (leftLabels.size() >= minSamplesLeaf && rightLabels.size() >= minSamplesLeaf) 
            {
                double gini = (leftLabels.size() * giniImpurity(leftLabels) + rightLabels.size() * giniImpurity(rightLabels)) / labels.size();
                if (gini < bestGini) 
                {
                    bestGini = gini;
                    bestFeature = i;
                    bestThreshold = threshold;
                }
            }
        }
    }
    if (bestFeature == -1) 
    {
        vector<int> counts(3, 0);
        for (int label : labels) 
        {
            counts[label]++;
        }
        TreeNode* node = new TreeNode();
        node->label = max_element(counts.begin(), counts.end()) - counts.begin();
        return node;
    }
    vector<vector<double>> leftFeatures, rightFeatures;
    vector<int> leftLabels, rightLabels;
    for (size_t i = 0; i < features.size(); ++i) 
    {
        if (features[i][bestFeature] <= bestThreshold) 
        {
            leftFeatures.push_back(features[i]);
            leftLabels.push_back(labels[i]);
        } else 
        {
            rightFeatures.push_back(features[i]);
            rightLabels.push_back(labels[i]);
        }
    }
    TreeNode* node = new TreeNode();
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftFeatures, leftLabels, depth + 1, maxDepth, minSamplesSplit, minSamplesLeaf);
    node->right = buildTree(rightFeatures, rightLabels, depth + 1, maxDepth, minSamplesSplit, minSamplesLeaf);
    return node;
}

int predict(TreeNode* node, const vector<double>& row) 
{
    if (node->label != -1) return node->label;
    if (row[node->featureIndex] <= node->threshold) 
    {
        return predict(node->left, row);
    } 
    else 
    {
        return predict(node->right, row);
    }
}

double crossValidate(const vector<vector<double>>& features, const vector<int>& labels, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int kFold) 
{
    int foldSize = features.size() / kFold;
    double totalAccuracy = 0;
    for (int fold = 0; fold < kFold; ++fold) 
    {
        vector<vector<double>> trainFeatures, testFeatures;
        vector<int> trainLabels, testLabels;
        for (int i = 0; i < features.size(); ++i) 
        {
            if (i >= fold * foldSize && i < (fold + 1) * foldSize) 
            {
                testFeatures.push_back(features[i]);
                testLabels.push_back(labels[i]);
            } 
            else 
            {
                trainFeatures.push_back(features[i]);
                trainLabels.push_back(labels[i]);
            }
        }
        TreeNode* tree = buildTree(trainFeatures, trainLabels, 0, maxDepth, minSamplesSplit, minSamplesLeaf);
        int correctPredictions = 0;
        for (size_t i = 0; i < testFeatures.size(); ++i) 
        {
            int prediction = predict(tree, testFeatures[i]);
            if (prediction == testLabels[i]) 
            {
                correctPredictions++;
            }
        }
        totalAccuracy += 1.0 * correctPredictions / testFeatures.size();
    }
    return totalAccuracy / kFold;
}

void printTree(TreeNode* node, int depth = 0) 
{
    if (!node) return;
    for (int i = 0; i < depth; ++i) cout << "    ";
    if (node->label != -1) 
    {
        cout << "Leaf: Label = " << node->label << el;
    } 
    else 
    {
        cout << "Feature[" << node->featureIndex << "] <= " << node->threshold << el;
        for (int i = 0; i < depth; ++i) cout << "    ";
        cout << "Left ->" << el;
        printTree(node->left, depth + 1);
        for (int i = 0; i < depth; ++i) cout << "    ";
        cout << "Right ->" << el;
        printTree(node->right, depth + 1);
    }
}

int main() 
{
    Faster;
    string trainFile = "train.txt";
    string testFile = "test.txt";
    string outputFile = "submission.csv";
    vector<vector<double>> trainFeatures, testFeatures;
    vector<int> trainLabels;
    readData(trainFile, trainFeatures, trainLabels, true);
    readData(testFile, testFeatures, trainLabels, false);
    int maxDepth = 5;
    int minSamplesSplit = 10;
    int minSamplesLeaf = 5;
    int kFold = 5;
    double accuracy = crossValidate(trainFeatures, trainLabels, maxDepth, minSamplesSplit, minSamplesLeaf, kFold);
    cout << "Cross" << accuracy << el;
    TreeNode* tree = buildTree(trainFeatures, trainLabels, 0, maxDepth, minSamplesSplit, minSamplesLeaf);
    //printTree(tree);
    ofstream outFile(outputFile);
    outFile << "ID,Label" << el; 
    int id = 1;
    for (const auto& row : testFeatures) 
    {
        int prediction = predict(tree, row);
        char label = (prediction == 0) ? 'L' : (prediction == 1) ? 'R' : 'B';
        outFile << id << "," << label << el;
        id++;
    }
    outFile.close();
    return 0;
}
