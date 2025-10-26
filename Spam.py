import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 0. 创建输出目录
os.makedirs("output", exist_ok=True)

# 1. 加载数据集
data = pd.read_csv("./data/SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])
data.columns = ['label', 'text']

# 2. 类别分布图
plt.figure(figsize=(5, 4))
sns.countplot(x='label', data=data, palette='Set2', legend=False)
plt.title("Message Class Distribution (ham vs spam)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("output/class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 4. 文本向量化
vec = CountVectorizer(stop_words='english')
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# 5. 训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. 模型预测与性能评估
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. 混淆矩阵热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. 词频最高的前 10 个词
word_counts = X_train_vec.toarray().sum(axis=0)
words = vec.get_feature_names_out()
freq_df = pd.DataFrame({'word': words, 'count': word_counts})
top10 = freq_df.sort_values(by='count', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x='count', y='word', data=top10, palette='viridis')
plt.title("Top 10 Most Frequent Words in Messages")
plt.xlabel("Word Count")
plt.ylabel("Word")
plt.savefig("output/top10_words.png", dpi=300, bbox_inches='tight')
plt.close()
