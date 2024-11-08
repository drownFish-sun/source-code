import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv
import scipy.sparse as sparse
import cvxpy as cp

from hyperg.utils import minmax_scale
from scipy.sparse import csr_matrix
import time
class HyperG:
    def __init__(self, objects=None, label=None, u=None, K=4, lamda=5, mu=0.05):
        self.construction(objects, label, u, K, lamda, mu)

    def cu_dis(self, objects):
        nt = len(objects)
        self.distances = np.zeros((nt, nt, 2))
        for i in range(nt):
            for j in range(i + 1, nt):
                manh_distance = self.test_manh(objects[i], objects[j])
                self.distances[i, j, 0] = manh_distance
                self.distances[j, i, 0] = manh_distance

                hog_distance = self.test_hog_distance(objects[i], objects[j])
                self.distances[i, j, 1] = hog_distance
                self.distances[j, i, 1] = hog_distance

    def test_manh(self, o1, o2):
        return sum(abs(a - b) for a, b in zip(o1, o2))
    def test_hog_distance(self, o1, o2):
        return distance.euclidean(o1, o2)
    def get_dis(self, a, b):
        return self.distances[a][b][0] + self.distances[a][b][1]
        # return self.distances[a][b][0]
    def H_construction(self, nt, K):
        m_neighbors = np.argpartition(self.distances[:, :, 0], kth=K+1, axis=1)
        m_neighbors_val = np.take_along_axis(self.distances[:, :, 0], m_neighbors, axis=1)
        m_neighbors = m_neighbors[:, :K+1]
        m_neighbors_val = m_neighbors_val[:, :K+1]
        for i in range(nt):
            if not np.any(m_neighbors[i, :] == i):
                m_neighbors[i, -1] = i
                m_neighbors_val[i, -1] = 0.
        node_idx = m_neighbors.reshape(-1)
        edge_idx = np.tile(np.arange(nt).reshape(-1, 1), (1, K+1)).reshape(-1)
        avg_dist = np.mean(self.distances[:, :, 0])
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))
        H_0 = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(nt, nt))

        m_neighbors = np.argpartition(self.distances[:, :, 1], kth=K+1, axis=1)
        m_neighbors_val = np.take_along_axis(self.distances[:, :, 1], m_neighbors, axis=1)
        m_neighbors = m_neighbors[:, :K+1]
        m_neighbors_val = m_neighbors_val[:, :K+1]
        for i in range(nt):
            if not np.any(m_neighbors[i, :] == i):
                m_neighbors[i, -1] = i
                m_neighbors_val[i, -1] = 0.
        node_idx = m_neighbors.reshape(-1)
        edge_idx = np.tile(np.arange(nt).reshape(-1, 1), (1, K+1)).reshape(-1)
        avg_dist = np.mean(self.distances[:, :, 1])
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))
        H_1 = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(nt, nt))

        H_0 = H_0.toarray()
        H_1 = H_1.toarray()
        H = np.hstack((H_0, H_1))
        return sparse.csr_matrix(H)
        # return H_1
    def calc_U(self, nt, label, u=None):
        self.U_ = np.zeros(nt)
        labels = np.unique(label)
        Sum_ = np.zeros(len(labels))
        for i in range(len(label)):
            for j in range(len(label)):
                if label[i] == label[j] and i != j:
                    self.U_[i] += self.get_dis(i, j)
        for i in range(len(label)):
            Sum_[label[i]] += self.U_[i]
        if Sum_[1] == 0:
            self.U_[np.where(label == 1)] = 1 / len(label[label == 1])
        Sum_[Sum_ == 0] = 1
        for i in range(len(label)):
            self.U_[i] = self.U_[i] * 10 / Sum_[label[i]]
        # for i in range(len(label)):
        #     self.U_[i] += u[i] * 10
        # self.U_[np.where(self.U_ == 0)] = self.U_[np.where(self.U_ != 0)].min()
        self.U_[len(label):] = self.U_[:len(label)].min() / 100
        self.U_ = minmax_scale(self.U_, (0.5, 2.0))
        # self.U_[np.where(label == 0)[0]] *= 1 / self.U_[np.where(label == 0)[0]].min()
        # self.U_[np.where(label == 1)[0]] *= 1 / self.U_[np.where(label == 1)[0]].min()
        # self.U_[len(label):] = self.U_[:len(label)].min() / 10
        # self.U_ /= 0.1
        # self.U_[:len(label)] *= (1 / self.U_[:len(label)].min())
        # print(self.U_)
    def init_Y(self, nt, label):
        labels = np.unique(label)
        Y = np.ones((nt, 2)) * (1 / len(labels))
        for i in range(len(label)):
            Y[i,] = 0
            Y[i, int(label[i])] = 1
        return Y

    def construction(self, objects=None, label=None, u=None, K=4, lamda=5, mu=0.05):
        self.label = label
        self.lamda = lamda
        self.mu = mu
        self.index = len(label)
        self.cu_dis(objects)
        self.objects = objects
        # 计算特征距离
        nt = len(objects)
        self.H = self.H_construction(nt, K)
        ne = self.H.shape[1]
        self.W = sparse.diags(np.ones(ne), shape=(ne, ne))
        # print(self.W)
        self.Y = self.init_Y(nt, label)
        self.calc_U(nt, label, u)
        # self.U_ = np.ones(nt)
        # self.U_ *= (1 / self.U_.min())
        self.U = sparse.diags(self.U_, shape=(nt, nt))
        # print(self.U)
        b = np.ones((ne, 1))
        c = np.ones((nt, 1))
        self.d = self.H.dot(self.W).dot(b).reshape(-1)
        self.delta = self.H.T.dot(self.U).dot(c).reshape(-1)
        self.dv = np.power(self.d, -0.5)
        self.de = np.power(self.delta, -1)
        # self.delta *= (K / self.delta.max())
        # print(self.delta)
        # print(self.d)
        self.invDe = sparse.diags(self.de, shape=(ne, ne))
        self.sqinvDv = sparse.diags(self.dv, shape=(nt, nt))
        self.Theta = self.sqinvDv.dot(self.U).dot(self.H).dot(self.W).dot(self.invDe).dot(self.H.T).dot(self.U).dot(self.sqinvDv)
        self.L2 = self.U - self.Theta
        # print(self.invDe)
        return HyperG
    def predict_(self):
        nt = self.H.shape[0]
        U = self.U
        H = self.H
        Y = self.Y
        W = self.W
        invDe = self.invDe
        sqinvDv = self.sqinvDv
        I = sparse.eye(nt)
        best_objective_value = 0.0
        Theta = self.Theta
        L = (I - (1 / self.lamda) * Theta)
        # L = I - (1 / (1 + self.lamda)) * Theta
        return inv(L.toarray()).dot(Y)
        # while True:
        #     Theta = sqinvDv.dot(U).dot(H).dot(W).dot(invDe).dot(H.T).dot(U).dot(sqinvDv)
        #     F = inv(I + 1 / self.lamda * (I - Theta)).dot(Y)
        #     # step = -np.transpose(F.T.dot(sqinvDv).dot(U).dot(H)).dot(
        #     #     np.transpose(invDe.dot(H.T).dot(U).dot(sqinvDv).dot(F))) + self.mu * 2 * W
        #     # dia = np.diag(step)
        #     # step_diag = np.zeros((ne, ne))
        #     # np.fill_diagonal(step_diag, dia)
        #     # dataset = W - 0.05 * step_diag
        #     # if dataset.min() < 0:
        #     #     continue
        #     # W = dataset
        #     W = np.transpose(F.T.dot(sqinvDv).dot(U).dot(H)).dot(
        #         np.transpose(invDe.dot(H.T).dot(U).dot(sqinvDv).dot(F))) / 2 * self.mu
        #     if W.min() < 0:
        #         break
        #     # self.W = W
        #     Dv = self.update(W, H)
        #     # self.Dv = Dv
        #     sqinvDv = np.sqrt(inv(Dv))
        #     Theta = sqinvDv.dot(U).dot(H).dot(W).dot(invDe).dot(H.T).dot(U).dot(sqinvDv)
        #     new_objective_value = np.linalg.det(F.T.dot(U - Theta).dot(F))
        #     new_objective_value += self.lamda * np.sum((F - Y) ** 2)
        #     new_objective_value += self.mu * np.trace(W.T.dot(W))
        #     if best_objective_value - new_objective_value > 1e-6 or best_objective_value == 0:
        #         best_objective_value = new_objective_value
        #     else:
        #         break
        # return F
    def predict(self):
        nt = self.H.shape[0]
        ne = self.H.shape[1]
        U = self.U
        H = self.H
        Y = self.Y
        W = self.W
        invDe = self.invDe
        sqinvDv = self.sqinvDv
        I = sparse.eye(nt)
        best_objective_value = 0.0
        iters = 5
        Theta = self.Theta
        for iter in range(iters):
            # L = (1 / self.lamda) * Theta + I
            L = I + (1 / self.lamda) * (I - Theta)
            F = inv(L.toarray()).dot(Y)
            W2 = invDe.dot(H.T).dot(U).dot(sqinvDv)
            W = np.transpose(F.T.dot(sqinvDv.toarray()).dot(U.toarray()).dot(H.toarray())).dot\
                (np.transpose(W2.toarray().dot(F))) / (self.mu * 2)
            if W.min() < 0:
                break
            new_objective_value = self.lamda * loss_F_Y(F, Y)
            new_objective_value += self.mu * np.trace(W.T.dot(W))
            if best_objective_value - new_objective_value > 1e-6 or best_objective_value == 0:
                best_objective_value = new_objective_value
            else:
                break
            w = [W[i, i] for i in range(ne)]
            self.W = sparse.diags(w, shape=(ne, ne))
            self.update(self.W, H)
            sqinvDv = self.sqinvDv
            W = self.W
            Theta = sqinvDv.dot(U).dot(H).dot(W).dot(invDe).dot(H.T).dot(U).dot(sqinvDv)
        W = sparse.csr_matrix(self.W)
        ar = [sqinvDv, U, H, W, invDe]
        self.Theta = sqinvDv.dot(U).dot(H).dot(W).dot(invDe).dot(H.T).dot(U).dot(sqinvDv)
        L = I + (1 / self.lamda) * (I - self.Theta)
        # L = (1 / self.lamda) * Theta + I
        self.L2 = U - self.Theta
        F = inv(L.toarray()).dot(Y)
        return F
    def update_W(self, F):
        nt = self.H.shape[0]
        ne = self.H.shape[1]
        U = self.U
        H = self.H
        invDe = self.invDe
        sqinvDv = self.sqinvDv
        W2 = invDe.dot(H.T).dot(U).dot(sqinvDv)
        W = np.transpose(F.T.dot(sqinvDv.toarray()).dot(U.toarray()).dot(H.toarray())).dot \
                (np.transpose(W2.toarray().dot(F))) / (self.mu * 2)
        w = [W[i, i] for i in range(ne)]
        self.W = sparse.diags(w, shape=(ne, ne))
        self.update(self.W, H)
        return

    def CVX(self):
        nt = self.H.shape[0]
        ne = self.H.shape[1]
        U = self.U
        H = self.H
        Y = self.Y
        W = self.W
        Dv = self.Dv
        De = self.De
        invDe = inv(De)
        sqinvDv = np.sqrt(inv(Dv))

        I = np.zeros((nt, nt), int)
        np.fill_diagonal(I, 1)
        previous_objective_value = 0.0
        while True:
            diag_tempW = cp.Variable(ne, nonneg=True)
            w_diag = [W[i, i] for i in range(ne)]
            diag_tempW.value = w_diag
            Theta = np.sqrt(inv(Dv)) @ U @ H @ self.toW(cp.diag(diag_tempW)) @ inv(De) @ H.T @ U @ np.sqrt(inv(Dv))
            F = inv(I + 1 / self.lamda * (I - Theta)) @ Y
            objective = cp.Minimize(cp.norm(
                F.T @ (U - np.sqrt(inv(Dv)) @ U @ H @ cp.diag(diag_tempW) @ inv(De) @ H.T @ U @ np.sqrt(inv(Dv))) @ F)
                                    + self.mu * cp.trace(self.toW(cp.diag(diag_tempW)).T @ self.toW(cp.diag(diag_tempW))))
            b = np.ones((ne, 1))
            d = H.dot(W).dot(b)
            constraints = [(H @ cp.diag(diag_tempW) @ b) - d <= 1e-6]
            # constraints.append(diag_tempW.value.min() > 0)
            problem = cp.Problem(objective, constraints)
            problem.solve()
            W = cp.diag(diag_tempW).value
            self.W = W
            # print(diag_tempW.value)
            new_objective_value = np.linalg.det(F.T @ (U - Theta) @ F)
            for i in range(F.shape[1]):
                diff = F[:, i] - Y[:, i]  # 计算向量之差
                distance_squared = np.dot(diff, diff)
                new_objective_value += distance_squared * self.lamda
            new_objective_value += self.mu * np.trace(self.toW(cp.diag(diag_tempW)).T @ self.toW(cp.diag(diag_tempW)))
            if previous_objective_value - new_objective_value > 1e-6 or previous_objective_value == 0:
                previous_objective_value = new_objective_value
            else:
                break
            # Dv = update(W, H)
        return F

    def toW(self, tempW):
        return tempW.value
    def update(self, W, H):
        b = np.ones((W.shape[1], 1))
        self.d = H.dot(W).dot(b).reshape(-1)
        self.sqinvDv = sparse.diags(np.power(self.d, -0.5), shape=(H.shape[0], H.shape[0]))
def loss_F_Y(F, Y):
    diff_F_Y = (F[:, 0] - F[:, 1]) * (Y[:, 0] - Y[:, 1])
    return np.sum(np.where(diff_F_Y < 0, diff_F_Y, 0)) * -1