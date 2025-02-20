## Dataset
**Source:** [Kaggle - Data for Admission in the University](https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university)

### Features:
- **GRE Score** (out of 340)
- **TOEFL Score** (out of 120)
- **University Rating** (out of 5)
- **Statement of Purpose (SOP)** (out of 5)
- **Letter of Recommendation Strength (LOR)** (out of 5)
- **Undergraduate GPA (CGPA)** (out of 10)
- **Research Experience** (either 0 or 1)
- **Chance of Admit** (ranging from 0 to 1)

## Steps in Analysis

### 1. Data Preprocessing
- Removed unnecessary columns ('Serial No.' and 'Chance of Admit').
- Binned numerical features into "High" and "Low" categories using histograms.
- Converted dataset into a list of transactions.

### 2. Encoding
- Used **TransactionEncoder** from `mlxtend` to transform categorical data into a one-hot encoded format.

### 3. Apriori Algorithm
- Extracted frequent itemsets using the **Apriori** algorithm.
- Applied a minimum support threshold of **0.3** to filter significant associations.

### 4. Association Rule Mining
- Generated association rules using the **Lift** metric with a **minimum threshold of 1**.
- Sorted rules by **confidence, lift, and support** to identify the most significant relationships.

## Key Findings
- High GRE and TOEFL scores, combined with high CGPA and research experience, strongly correlate with a **high chance of admission**.
- SOP and LOR, while important, play a lesser role compared to numerical scores and research experience.
