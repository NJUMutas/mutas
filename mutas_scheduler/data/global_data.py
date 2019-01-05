import numpy as np


class GlobalData:
    EPS = 2.2204e-16  # 数学常数

    # 算法输入数据
    USER_NUM = 5  # 用户数量
    SERVER_NUM = 4  # 服务器数量
    TRANS_NUM = 3  # 每个用户的任务数量

    FUNCTION_MASK = np.mat([  # 指示某一server上是否有某一功能 SERVER_NUM * TRANS_NUM
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ])

    SERVER_CAPACITY = np.array([200, 100, 300, 200])  # 各个服务器容量

    LINK_LATENCY = np.mat([  # 各个服务器间连接延迟 SERVER_NUM * SERVER_NUM
        [0, 0.02, 0.01, 0.03],
        [0.02, 0, 0.05, 0.03],
        [0.01, 0.05, 0, 0.03],
        [0.03, 0.03, 0.03, 0]
    ])

    AP_SERVER_LATENCY = np.zeros(shape=[USER_NUM, SERVER_NUM])  # 服务器和AP之间的延迟 USER_NUM * SERVER_NUM

    edge_server = 2  # 边缘计算服务器数量
    fog_server = 1  # 雾计算服务器数量
    cloud_server = 1  # 云计算服务器数量

    DATA_SIZE = np.zeros(shape=USER_NUM)
    USER_TRANS = np.zeros(shape=USER_NUM)

    NEW_FUNCTION_MASK = []
    NEW_DATA_SIZE = np.zeros(shape=USER_NUM * SERVER_NUM * TRANS_NUM)
    TEMP_NEW_LATENCY = np.zeros(shape=USER_NUM * TRANS_NUM)
    NEW_LATENCY = None

    C = None
    b = np.zeros(shape=USER_NUM * TRANS_NUM)

    def __init__(self):

        self.random_generate()  # 随机生成数据
        self.check()  # 检测输入数据不一致性
        self.new()  # 更新数据

    def update(self, data):
        for k, v in data.items():
            setattr(self, k, int(v))
        self.check()
        self.new()

    def random_generate(self):
        # 随机生成 AP_SERVER_LATENCY
        for i in range(self.USER_NUM):
            edge_latency = np.random.random_integers(low=20, high=80, size=self.edge_server) / 1000
            fog_latency = np.random.random_integers(low=50, high=90, size=self.fog_server) / 1000
            cloud_latency = np.random.random_integers(low=100, high=300, size=self.cloud_server) / 1000
            latency = np.hstack((edge_latency, fog_latency, cloud_latency))
            self.AP_SERVER_LATENCY[i, :] = latency

        # 随机生成用户数据 DATA_SIZE，USER_TRANS
        self.DATA_SIZE = np.random.random_integers(low=1, high=5, size=self.USER_NUM)
        self.USER_TRANS = np.random.random_integers(low=0, high=self.TRANS_NUM - 1, size=self.USER_NUM)

    def new(self):

        # FUNCTION_MASK => NEW_FUNCTION_MASK
        for i in range(self.SERVER_NUM):
            for j in range(self.TRANS_NUM):
                temp = self.FUNCTION_MASK[i, j]
                temp_matrix = np.ones(shape=self.USER_NUM) * temp
                self.NEW_FUNCTION_MASK = np.hstack((self.NEW_FUNCTION_MASK, temp_matrix))

        # DATA_SIZE => NEW_DATA_SIZE
        for i in range(self.USER_NUM):
            for j in range(self.SERVER_NUM):
                for k in range(self.TRANS_NUM):
                    index = j * self.USER_NUM * self.TRANS_NUM + k * self.USER_NUM + i
                    if k <= self.USER_TRANS[i]:
                        self.NEW_DATA_SIZE[index] = self.DATA_SIZE[i]
                    else:
                        self.NEW_DATA_SIZE[index] = 0

        # TEMP_NEW_LATENCY
        for i in range(self.TRANS_NUM):
            for j in range(self.USER_NUM):
                index = i * self.USER_NUM + j
                if i == 1 or i == self.USER_TRANS[j]:
                    self.TEMP_NEW_LATENCY[index] = 1

        # NEW_LATENCY
        for i in range(self.SERVER_NUM):
            temp_B = np.repeat(self.AP_SERVER_LATENCY[:, i], self.TRANS_NUM) * self.TEMP_NEW_LATENCY
            if self.NEW_LATENCY is None:
                self.NEW_LATENCY = temp_B
            else:
                self.NEW_LATENCY = np.hstack((self.NEW_LATENCY, temp_B))

        # C
        for i in range(self.SERVER_NUM):
            temp = np.diag(
                self.NEW_FUNCTION_MASK[i * self.USER_NUM * self.TRANS_NUM: (i + 1) * self.USER_NUM * self.TRANS_NUM])
            if self.C is None:
                self.C = temp
            else:
                self.C = np.hstack((self.C, temp))

        # b
        for i in range(self.TRANS_NUM):
            for j in range(self.USER_NUM):
                index = i * self.USER_NUM + j
                if i <= self.USER_TRANS[j]:
                    self.b[index] = 1

    def check(self):
        error = False
        if self.FUNCTION_MASK.shape[0] != self.SERVER_NUM:
            error = True

        if self.SERVER_CAPACITY.shape[0] != self.SERVER_NUM:
            error = True

        if self.LINK_LATENCY.shape[0] != self.SERVER_NUM:
            error = True

        if self.edge_server + self.fog_server + self.cloud_server != self.SERVER_NUM:
            error = True

        if error:
            print("data check error")
            raise Exception("data check error")
        else:
            print("data check pass")