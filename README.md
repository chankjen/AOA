### **Part 1: Attribute Oriented Analysis (AOA)**

#### **1.1 Definition and Purpose**

**Attribute Oriented Analysis (AOA)**, also known as **Attribute Oriented Induction (AOI)**, is a data analysis and knowledge discovery technique. Its primary purpose is to **generalize specific, low-level data into more abstract, high-level concepts**.

Think of it as a "bottom-up" summarization process. Instead of looking at individual data points (e.g., "John, age 23, salary $52,000"), AOA groups them into meaningful categories (e.g., "Young Adults, Medium Income").

#### **1.2 Key Concepts**

*   **Attribute:** A feature or column in your dataset (e.g., `Age`, `Salary`, `City`).
*   **Concept Hierarchy:** A predefined tree-like structure that organizes data from specific values to general concepts.
    *   **Example for `Age`:**
        *   Level 0 (Raw Data): `23`, `24`, `67`, `68`
        *   Level 1 (Generalized): `20-29`, `60-69`
        *   Level 2 (More Generalized): `Young`, `Senior`
        *   Level 3 (Very Generalized): `Adult`
*   **Generalization:** The process of replacing a low-level concept (e.g., "Toronto") with a higher-level one (e.g., "Canada"). This is done by "climbing" the concept hierarchy.
*   **Threshold Control:** A mechanism to stop generalization. It prevents over-generalization where the data becomes too vague to be useful. Common thresholds are:
    *   **Attribute Threshold (`a_thresh`)**: The maximum number of distinct values allowed for an attribute after generalization.
    *   **Relation Threshold (`r_thresh`)**: The maximum number of tuples (rows) allowed in the final generalized relation.

#### **1.3 The AOA Algorithm (Step-by-Step)**

1.  **Collect the Relevant Data:** Start with the initial relation (table) of interest.
2.  **Preprocess Data:** Handle missing values, noise, etc. (This is where our practicals begin).
3.  **Select Attributes for Analysis:** Decide which attributes to generalize.
4.  **Perform Generalization:**
    *   For each selected attribute, check if the number of distinct values exceeds the attribute threshold (`a_thresh`).
    *   If it does, replace the values with the next higher-level concept from its hierarchy.
    *   Repeat this process until the number of distinct values for that attribute is <= `a_thresh`.
5.  **Aggregate and Summarize:**
    *   After generalizing all attributes, identical tuples will appear.
    *   Merge these identical tuples.
    *   For quantitative attributes (like `Salary`), you might add a `count` column to show how many original tuples were merged. You can also compute aggregates like `average_salary`.
6.  **Present the Generalized Relation:** The result is a compact, summarized table that represents the core patterns in the data.

#### **1.4 Advantages of AOA**

*   **Data Reduction:** Drastically reduces the size of the data, making it easier to analyze.
*   **Pattern Discovery:** Reveals macro-level trends and patterns that are invisible in the raw data.
*   **Background Knowledge Integration:** Concept hierarchies allow the incorporation of domain expertise.
*   **Multiple-Level Mining:** Allows users to drill down or roll up through different levels of generalization to see data at various abstraction levels.

#### **1.5 AOA vs. Alternative Methods**

*   **AOA vs. Simple Aggregation (GROUP BY):** AOA is more powerful because it uses concept hierarchies to create semantically meaningful groups, not just syntactic ones. Grouping by "Young" and "Senior" is more insightful than grouping by raw age numbers.
*   **AOA vs. OLAP (Online Analytical Processing):** They are very similar in spirit. OLAP operations (roll-up, drill-down) are the interactive equivalent of performing AOA. AOA can be seen as the algorithmic foundation for OLAP's roll-up operation.

---

### **Part 2: Practicals - The Complete Data Mining Workflow**

This practical session will walk you through a complete example, from raw data to knowledge visualization.

#### **Scenario: Analyzing a Customer Dataset**

We want to understand the general profile of customers based on their age, salary, and city.

**Step 1: Data Preprocessing**

**Task:** Load the data, handle missing values, and ensure data types are correct.

```python
import pandas as pd
import numpy as np

# Sample Raw Data (with intentional issues)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Age': [23, 23, 67, 32, 19, 21, 67, None],  # Missing value
    'Salary': [52000, 48000, 80000, 55000, 21000, 45000, 82000, 60000],
    'City': ['Toronto', 'Mississauga', 'New York', 'Toronto', 'Boston', 'Boston', 'New York', 'Waterloo']
}

df = pd.DataFrame(data)
print("--- Raw Data ---")
print(df)
print("\n--- Info ---")
print(df.info())
```

**Issues Identified:** One missing value in the `Age` column.

**Preprocessing Code:**

```python
# 1. Handle Missing Values: Fill with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)
df['Age'] = df['Age'].astype(int) # Convert to integer

# 2. Verify the cleanup
print("\n--- Cleaned Data ---")
print(df)
print(f"\nMissing values?\n{df.isnull().sum()}")
```

**Step 2: Knowledge Representation (Defining Concept Hierarchies)**

**Task:** We will represent our knowledge as dictionaries that map specific values to generalized concepts.

```python
# Define Concept Hierarchies
# Hierarchy for 'Age'
def generalize_age(age):
    if age < 25:
        return 'Young'
    elif age < 60:
        return 'Middle-Aged'
    else:
        return 'Senior'

# Hierarchy for 'Salary'
def generalize_salary(salary):
    if salary < 30000:
        return 'Low'
    elif salary < 70000:
        return 'Medium'
    else:
        return 'High'

# Hierarchy for 'City' (Geography)
city_to_country = {
    'Toronto': 'Canada',
    'Mississauga': 'Canada',
    'Waterloo': 'Canada',
    'New York': 'USA',
    'Boston': 'USA'
}

# Apply the generalizations to create new columns
df['Age_Group'] = df['Age'].apply(generalize_age)
df['Income_Level'] = df['Salary'].apply(generalize_salary)
df['Country'] = df['City'].map(city_to_country)

print("\n--- Data with Generalized Attributes ---")
print(df[['CustomerID', 'Age', 'Age_Group', 'Salary', 'Income_Level', 'City', 'Country']])
```

**Step 3: Attribute Oriented Induction (The AOA Algorithm)**

**Task:** We will now perform the core AOA steps: generalization and aggregation.

```python
# We are no longer interested in the original low-level attributes.
# Let's select only the generalized attributes for analysis.
generalized_df = df[['Age_Group', 'Income_Level', 'Country']].copy()

# This is the AOA aggregation step (similar to a powerful GROUP BY).
# We count the number of original tuples merged into each generalized tuple.
aoa_result = generalized_df.groupby(['Age_Group', 'Income_Level', 'Country']).size().reset_index(name='Count')

print("\n--- AOA Result (Generalized Relation) ---")
print(aoa_result)
```

**Interpretation of the Result:**
The table shows the summarized knowledge. For example, one row might show: `(Young, Medium, Canada), Count: 2`. This tells us that there are two young customers from Canada with a medium income level. This is a much more comprehensible piece of knowledge than looking at the raw data.

**Step 4: Knowledge Visualization**

**Task:** Create visualizations to make the discovered patterns even clearer.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Bar Plot: Count by Age Group and Income Level
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(data=aoa_result, x='Age_Group', y='Count', hue='Income_Level', palette='viridis')
plt.title('Customer Count by Age and Income')
plt.legend(title='Income Level')

# 2. Stacked Bar Chart: Distribution across Countries
plt.subplot(1, 2, 2)
# Pivot the data for a stacked bar chart
pivot_df = aoa_result.pivot_table(index='Age_Group', columns='Country', values='Count', fill_value=0)
pivot_df.plot(kind='bar', stacked=True, ax=plt.gca()) # plt.gca() gets the current axis
plt.title('Customer Count by Age and Country')
plt.ylabel('Count')
plt.legend(title='Country')
plt.tight_layout()
plt.show()

# 3. Treemap (Excellent for hierarchical data)
import squarify # You may need to install: pip install squarify

plt.figure(figsize=(12, 8))
# Create labels for the treemap: "AgeGroup-IncomeLevel (Count)"
labels = aoa_result.apply(lambda row: f"{row['Age_Group']}-{row['Income_Level']}\n({row['Count']})", axis=1)
sizes = aoa_result['Count']
colors = plt.cm.Set3(range(len(sizes)))

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
plt.axis('off')
plt.title('Customer Segments (Treemap) - Area represents Count')
plt.show()
```

#### **Conclusion of the Practical**

You have successfully completed the entire data mining pipeline:
1.  **Preprocessing:** Cleaned the raw data.
2.  **Knowledge Representation:** Defined concept hierarchies that encapsulate domain knowledge.
3.  **Attribute Oriented Analysis:** Applied the hierarchies to generalize the data and aggregated it to find high-level patterns.
4.  **Visualization:** Used bar charts and treemaps to present the discovered knowledge in an intuitive, visual format.

The final output is no longer a list of customers but a summary of customer *segments*, which is far more useful for strategic decision-making.
