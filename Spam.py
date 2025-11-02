import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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

# 7. 绘制混淆矩阵热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. 统计训练集中出现频率最高的前10个词
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

# 9. 单条短信分类过程可视化（改进版）
# 选择测试集中一条短信
sample_idx = 53  # 你可以改成任意测试集索引
sample_text = X_test.iloc[sample_idx]
sample_label = y_test.iloc[sample_idx]
sample_vec = vec.transform([sample_text])  # 1 x V 矩阵（词频）

# 特征名与模型参数
feature_names = vec.get_feature_names_out()
log_prob = model.feature_log_prob_  # shape (n_classes, n_features)
classes = model.classes_
class_log_prior = model.class_log_prior_

# 从稀疏向量获取出现的词索引与对应计数（保证包含全部在短信中出现的词，包括重复）
sample_counts = sample_vec.toarray().ravel()  # length V
word_indices = np.where(sample_counts > 0)[0]
words_in_msg = feature_names[word_indices]
counts_in_msg = sample_counts[word_indices].astype(int)

# 计算每个词对 ham 与 spam 的贡献（count * log P(word|class)）
# 找到类索引（确保正确对应 class_log_prior）
class_to_index = {c: i for i, c in enumerate(classes)}
# 提取对应行
# 结果数组与 words_in_msg 对应
ham_idx = class_to_index.get('ham', None)
spam_idx = class_to_index.get('spam', None)
# 若某个类别名不存在（极少见），则用类索引 0/1 对应
if ham_idx is None or spam_idx is None:
    # 保险处理：若类标签不是 'ham'/'spam'，使用 classes 的两个元素按索引0、1
    ham_idx = 0
    spam_idx = 1

ham_word_log = log_prob[ham_idx, word_indices]
spam_word_log = log_prob[spam_idx, word_indices]

ham_contrib = counts_in_msg * ham_word_log
spam_contrib = counts_in_msg * spam_word_log

# 构建 DataFrame，保留原短信中词的出现顺序
df_contrib = pd.DataFrame({
    'word': words_in_msg,
    'count': counts_in_msg,
    'ham_logprob_per_occurrence': ham_word_log,
    'spam_logprob_per_occurrence': spam_word_log,
    'ham_contrib': ham_contrib,
    'spam_contrib': spam_contrib
})
# 计算差值（spam - ham），并按差值绝对值排序以便查看最有影响力的词
df_contrib['diff'] = df_contrib['spam_contrib'] - df_contrib['ham_contrib']
df_contrib_sorted = df_contrib.sort_values(by='diff', ascending=False).reset_index(drop=True)

# 保存表格，方便外部查看
df_contrib_sorted.to_csv("output/word_contributions.csv", index=False, encoding='utf-8-sig')

# 计算最终类得分（log space），使用每个词的贡献之和 + 类先验
ham_score = class_log_prior[ham_idx] + df_contrib['ham_contrib'].sum()
spam_score = class_log_prior[spam_idx] + df_contrib['spam_contrib'].sum()

# 用数值稳定的 softmax（log-sum-exp）
scores = np.array([ham_score, spam_score])
max_score = np.max(scores)
exp_scores = np.exp(scores - max_score)
probs = exp_scores / exp_scores.sum()

# 输出到控制台，便于快速查看
print("\n=== Single Message Analysis (improved) ===")
print("Message Text:", sample_text)
print("True Label:", sample_label)
print(f"Ham score (log): {ham_score:.4f}")
print(f"Spam score (log): {spam_score:.4f}")
print(f"Ham probability: {probs[0]:.6f}")
print(f"Spam probability: {probs[1]:.6f}")
predicted = ['ham', 'spam'][int(np.argmax(scores) == 1)] if len(classes) >= 2 else classes[np.argmax(scores)]
print("Predicted Label:", predicted)

# 绘图：每个单词对两个类的贡献对比（显示全部单词）
n_words = len(df_contrib_sorted)
fig_h = max(4, 0.4 * n_words + 2)  # 根据单词数量自适应高度
plt.figure(figsize=(10, fig_h))
# 为了排版工整，横坐标显示贡献值，纵坐标按 diff 排序显示单词
order = df_contrib_sorted['word']
# 将 DataFrame melt 成长格式以便 seaborn 画对比条形
df_melt = df_contrib_sorted.melt(id_vars=['word'], value_vars=['ham_contrib', 'spam_contrib'],
                                 var_name='Class', value_name='Contribution')
# 调整 Class 名
df_melt['Class'] = df_melt['Class'].map({'ham_contrib': 'ham', 'spam_contrib': 'spam'})
sns.barplot(x='Contribution', y='word', hue='Class', data=df_melt, order=order, dodge=True)
plt.title("Each Word's Contribution to log P(message|class) (all words shown)")
plt.xlabel("Contribution (count * log P(word|class))")
plt.ylabel("Word (ordered by spam-ham difference)")
plt.legend(title="Class")
plt.tight_layout()
plt.savefig("output/word_contributions.png", dpi=300, bbox_inches='tight')
plt.close()

# 绘图：累计堆叠贡献，展示如何从先验到最终得分累积（按排序顺序累积）
# 为可视化累积效果，我们按 df_contrib_sorted 的顺序累加 spam_contrib - ham_contrib，
# 并画出两个类的累积得分（先验 + 累积）
cum_ham = class_log_prior[ham_idx] + np.cumsum(df_contrib_sorted['ham_contrib'].values)
cum_spam = class_log_prior[spam_idx] + np.cumsum(df_contrib_sorted['spam_contrib'].values)
xlabels = df_contrib_sorted['word'].astype(str).tolist()
# 若词太多只显示部分 x 标签，避免挤压
plt.figure(figsize=(12, max(4, 0.35 * n_words + 2)))
plt.plot(range(1, n_words+1), cum_ham, marker='o', label='cum ham score (log prior + contributions)')
plt.plot(range(1, n_words+1), cum_spam, marker='o', label='cum spam score (log prior + contributions)')
plt.xticks(range(1, n_words+1), xlabels, rotation=45, ha='right')
plt.xlabel('Words (in order of spam-ham diff)')
plt.ylabel('Cumulative log score')
plt.title('Cumulative Scores as Each Word is Added (shows decision formation)')
plt.legend()
plt.tight_layout()
plt.savefig("output/cumulative_contributions.png", dpi=300, bbox_inches='tight')
plt.close()

# 最终概率柱状图（保证类别顺序与 probs 一致）
plt.figure(figsize=(5, 4))
# 使用 classes 顺序：我们想按 ['ham','spam'] 显示概率
display_classes = ['ham', 'spam']
display_probs = [probs[0], probs[1]]
sns.barplot(x=display_classes, y=display_probs, palette='Set2')
plt.title("Final Prediction Probabilities")
plt.ylabel("Probability")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.savefig("output/final_prediction.png", dpi=300, bbox_inches='tight')
plt.close()

