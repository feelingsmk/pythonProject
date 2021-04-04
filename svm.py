import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split


def get_pre_data(pre_wave_path):
    data = np.genfromtxt(pre_wave_path, delimiter=',')
    x = data[:, 1:]  # 数据特征
    # print(x)
    y = data[:, 0].astype(int)  # 标签
    # print("数据标签：{}".format(y))
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    return x_std, y


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:, 1:]  # 数据特征
    y = data[:, 0].astype(int)  # 标签
    # print(y)
    scaler = StandardScaler()
    #     x_std = x
    x_std = scaler.fit_transform(x)  # 标准化
    # 将数据划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重,非线性多维支持向量分类
    svc = SVC(class_weight='balanced', probability=True)
    c_range = np.logspace(-5, 15, 11, base=2)  # 2的-5次幂到2的13次幂的50个数，默认基数是10
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围
    param_grid = [{'kernel': ['rbf', ''], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    grid.fit(x_train, y_train)
    # 计算测试集精度
    a, c = get_pre_data('pre_datafile.csv')
    # a,b,c,d = load_data('pre_datafile.csv')
    print(grid.predict(a))
    # print(grid.predict_proba(a))
    print(grid.score(a, c))

    print('最佳参数是:{}'.format(grid.best_params_))
    print('最佳分数是:{}'.format(grid.best_score_))
    print('最佳模型是:{}'.format(grid.best_estimator_))
    # print("训练集精度:{}".format(grid.score(x_train, y_train)))  # 精度
    # print("测试集精度:{}".format(grid.score(x_test, y_test)))
    return grid.best_estimator_, x_test, x_train, y_test, y_train


if __name__ == '__main__':
    aaa, ccc, ddd, CCC, DDD = svm_c(*load_data('datafile2.csv'))

