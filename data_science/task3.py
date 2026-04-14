import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体以支持图表显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 步骤 1: 数据准备 (由于篇幅限制，这里提取了部分诗句作为核心训练集)
# 实际数据科学项目中，需通过爬虫或文本库加载完整诗集
# ==========================================
data = {
    '杜甫': [
        "两个黄鹂鸣翠柳", "一行白鹭上青天", "窗含西岭千秋雪", "门泊东吴万里船", # 绝句
        "国破山河在", "城春草木深", "感时花溅泪", "恨别鸟惊心", # 春望
        "好雨知时节", "当春乃发生", "随风潜入夜", "润物细无声", # 春夜喜雨
        "昔闻洞庭水", "今上岳阳楼", "吴楚东南坼", "乾坤日夜浮", # 登岳阳楼
        "细草微风岸", "危樯独夜舟", "星垂平野阔", "月涌大江流", # 旅夜书怀
        "功盖三分国", "名成八阵图", "江流石不转", "遗恨失吞吴"  # 八阵图
    ],
    '李白': [
        "床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", # 静夜思
        "危楼高百尺", "手可摘星辰", "不敢高声语", "恐惊天上人", # 夜宿山寺
        "众鸟高飞尽", "孤云独去闲", "相看两不厌", "只有敬亭山", # 独坐敬亭山
        "耶溪采莲女", "见客棹歌回", "笑畏郎猜测", "娇羞不肯来", # 越女词
        "天下伤心处", "劳劳送客亭", "春风知别苦", "不遣柳条青", # 劳劳亭
        "秋浦长似秋", "萧条使人愁", "客愁不可度", "行上东大楼"  # 秋浦歌
    ]
}

# 构建 DataFrame
rows = []
for author, lines in data.items():
    for i, line in enumerate(lines):
        # 记录：文本，作者，在原诗中的相对位置（第1句=0，第2句=1...以此类推，取模4或8计算）
        # 这里为了简化，假设以4句为一首绝句/律诗的一半进行位置标注
        position = (i % 4) + 1 
        rows.append({'text': line, 'author': author, 'position': position})

df = pd.DataFrame(rows)

# 目标测试句
test_sentence = ["白水绕东城"]

# ==========================================
# 步骤 2: 特征工程 (TF-IDF 与 N-Gram)
# 提取字级别特征，ngram_range=(1,2) 捕捉单字和双字词
# ==========================================
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df['text'])
y_author = df['author']
y_position = df['position']

X_test = vectorizer.transform(test_sentence)

# ==========================================
# 步骤 3: 统计判别模型建立与预测
# ==========================================
# 任务 1：判别作者
clf_author = MultinomialNB(alpha=0.1)
clf_author.fit(X_train, y_author)
pred_author = clf_author.predict(X_test)[0]
prob_author = clf_author.predict_proba(X_test)[0]
classes_author = clf_author.classes_

# 任务 2：判别句子位置 (第几句)
clf_position = MultinomialNB(alpha=0.1)
clf_position.fit(X_train, y_position)
pred_position = clf_position.predict(X_test)[0]
prob_position = clf_position.predict_proba(X_test)[0]
classes_position = clf_position.classes_

# ==========================================
# 步骤 4: 结果输出与数据可视化
# ==========================================
print(f"目标诗句: '{test_sentence[0]}'")
print(f"1) 作者判决结果: {pred_author}")
print(f"2) 位置判决结果: 第 {pred_position} 句\n")

# 绘图设置 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图 1: 特征空间降维可视化 (PCA) - 展示数据科学的中间过程
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['author'], palette='Set1', ax=axes[0], s=80, alpha=0.7)
axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], color='gold', marker='*', s=300, edgecolor='black', label='测试句: 白水绕东城')
axes[0].set_title('TF-IDF特征空间 PCA降维分布图')
axes[0].set_xlabel('主成分 1')
axes[0].set_ylabel('主成分 2')
axes[0].legend()

# 图 2: 贝叶斯统计概率分布条形图
probs = list(prob_author) + list(prob_position)
labels = [f"作者:{c}" for c in classes_author] + [f"位置:第{p}句" for p in classes_position]
colors = ['#ff9999', '#66b3ff'] + ['#99ff99']*len(classes_position)

axes[1].bar(labels, probs, color=colors, edgecolor='black')
axes[1].set_title('目标句的统计后验概率分布 (Bayes Posterior)')
axes[1].set_ylabel('概率 P(c|x)')
axes[1].tick_params(axis='x', rotation=45)

for i, v in enumerate(probs):
    axes[1].text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()