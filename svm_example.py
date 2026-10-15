"""线性 SVM（软间隔，hinge loss + 次梯度下降）及 k 折交叉验证选 C 的示例实现。

逐行对应伪代码中的 trainSVM / crossValidateSVM，并补上伪代码里用到但未定义的 predictSVM。
运行: python svm_example.py
"""
import numpy as np


def train_svm(X, y, C, learning_rate, n_iterations):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(n_iterations):
        for i in range(n_samples):
            margin = y[i] * (np.dot(X[i], w) + b)

            if margin >= 1:
                # 没有违反间隔：只有正则项梯度
                grad_w = w
                grad_b = 0.0
            else:
                # 违反间隔：正则项 + hinge loss 的次梯度
                grad_w = w - C * y[i] * X[i]
                grad_b = -C * y[i]

            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b

    return w, b


def predict_svm(X, w, b):
    # 决策函数取符号，0 归为 +1，保证输出只有 -1 / 1
    return np.where(X @ w + b >= 0, 1, -1)


def cross_validate_svm(X, y, C_values, k, learning_rate, n_iterations, seed=0):
    n_samples = X.shape[0]
    # 先打乱再切成 k 折，避免数据按类别排序时某一折只有一个类
    indices = np.random.default_rng(seed).permutation(n_samples)
    folds = np.array_split(indices, k)

    best_C = None
    best_score = -np.inf

    for C in C_values:
        scores = []
        for fold in range(k):
            val_idx = folds[fold]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold])

            w, b = train_svm(X[train_idx], y[train_idx], C, learning_rate, n_iterations)
            y_pred = predict_svm(X[val_idx], w, b)
            scores.append(np.mean(y_pred == y[val_idx]))

        avg_score = np.mean(scores)
        print(f"  C = {C:<7} 每折准确率 = {np.round(scores, 3)}  平均 = {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_C = C

    return best_C


def make_data(n_per_class, rng):
    """两团部分重叠的二维高斯点，标签为 -1 / 1。"""
    X_pos = rng.normal(loc=[2.0, 2.0], scale=1.2, size=(n_per_class, 2))
    X_neg = rng.normal(loc=[-1.0, -1.0], scale=1.2, size=(n_per_class, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_per_class), -np.ones(n_per_class)]).astype(int)
    return X, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1. 构造数据并划分训练集 / 测试集
    X, y = make_data(n_per_class=100, rng=rng)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    X_train, y_train = X[:160], y[:160]
    X_test, y_test = X[160:], y[160:]

    # 2. 标准化（只用训练集的均值和方差）
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 3. 在训练集上做 5 折交叉验证选 C
    C_values = [0.001, 0.01, 0.1, 1, 10]
    learning_rate = 0.001
    n_iterations = 100
    print("交叉验证结果:")
    best_C = cross_validate_svm(X_train, y_train, C_values, k=5,
                                learning_rate=learning_rate, n_iterations=n_iterations)
    print(f"\n最佳 C = {best_C}")

    # 4. 用最佳 C 在整个训练集上重新训练，并在测试集上评估
    w, b = train_svm(X_train, y_train, best_C, learning_rate, n_iterations)
    test_acc = np.mean(predict_svm(X_test, w, b) == y_test)
    print(f"w = {np.round(w, 4)}, b = {b:.4f}")
    print(f"测试集准确率 = {test_acc:.4f}")

    # 5. 几个手工样本（按原始坐标给出，需要同样标准化）
    samples = np.array([[3.0, 3.0], [-2.0, -2.0], [0.5, 0.5]])
    preds = predict_svm((samples - mean) / std, w, b)
    for s, p in zip(samples, preds):
        print(f"样本 {s} -> 预测类别 {p:+d}")
