import numpy as np
import pandas as pd
import ast

from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings(action='ignore')

genre = pd.read_json('genre_gn_all.json', typ='series')
meta = pd.read_json('song_meta.json', encoding='utf-8')
train = pd.read_json('train.json', encoding='utf-8')
valid = pd.read_json('val.json', encoding='utf-8')
test = pd.read_json('test.json', encoding='utf-8')

# train 데이터의 title을 형태소 분석 한 데이터 별도 활용(Windows 상 khaiii 사용 불가능)
decom_train = pd.read_csv('train_title.csv', encoding='utf-8')
decom_train.columns = ['_', 'id', 'keywords']
decom_train['keywords'] = decom_train['keywords'].apply(lambda x:ast.literal_eval(x))

train = train[train['songs'].apply(lambda x:len(x)==200)]  # songs가 200개인 데이터만을 사용
train = train[:3000]                                     # 3000개 학습 데이터 샘플 사용
n_songs = 500                                             # 최다 등장 500개 songs만을 사용
n_titles = 200                                           # 최다 등장 200개 title 형태소만을 사용
undup_songs = np.array((pd.Series(np.concatenate(train['songs'].values)).value_counts()[:n_songs].index))
undup_gnr = np.unique(np.concatenate(meta['song_gn_gnr_basket']))
undup_dtl_gnr = np.unique(np.concatenate(meta['song_gn_dtl_gnr_basket']))
undup_title = np.array((pd.Series(np.concatenate(decom_train['keywords'].values)).value_counts()[:n_titles].index))
enc = MultiLabelBinarizer()
enc_gnr = MultiLabelBinarizer()
enc_dtl_gnr = MultiLabelBinarizer()
enc_title = MultiLabelBinarizer()
enc.fit([undup_songs])
enc_gnr.fit([undup_gnr])
enc_dtl_gnr.fit([undup_dtl_gnr])
enc_title.fit([undup_title])


class MF():
    def __init__(self, rating_mat, dim_latent, l2, alpha, l_rate, n_epochs):
        self.r_mat = rating_mat
        self.n_users, self.n_items = rating_mat.shape
        self.dim_latent = dim_latent
        self.l2 = l2
        self.alpha = alpha
        self.l_rate = l_rate
        self.n_epochs = n_epochs

    def train(self):
        # latent matrix 초기화
        self.user_mat = np.random.normal(size=(self.n_users, self.dim_latent))
        self.item_mat = np.random.normal(size=(self.n_items, self.dim_latent))
        print(f'user matrix shape : {self.user_mat.shape}')
        print(f'item matrix shape : {self.item_mat.shape}')

        # error 값 기록
        p_err_list = []
        c_err_list = []
        reg_list = []
        loss_list = []

        for i in range(self.n_epochs):
            print(f'epoch {i + 1} / {self.n_epochs}')
            # 분해 행렬 dot-product
            pred = self.user_mat.dot(self.item_mat.T)

            # binary rating matrix & confidence matrix 생성
            p_mat = np.copy(self.r_mat)
            p_mat[p_mat > 0] = 1
            c_mat = 1 + self.alpha * self.r_mat

            p_err, c_err, reg, loss = self.ALS_loss(self.r_mat, pred, c_mat, p_mat)
            p_err_list.append(p_err)
            c_err_list.append(c_err)
            reg_list.append(reg)
            loss_list.append(loss)

            self.optim_user(c_mat, p_mat)
            self.optim_item(c_mat, p_mat)

        final_pred = self.user_mat.dot(self.item_mat.T)
        return final_pred, p_err_list, c_err_list, reg_list, loss_list

    def ALS_loss(self, r_mat, p_mat, c_mat, pred):
        p_err = np.square(p_mat - pred)
        c_err = np.sum(c_mat * p_err)
        # weighted matrix factorization with regularization value(l2)
        reg = self.l2 * (np.sum(np.square(self.user_mat)) + np.sum(np.square(self.item_mat)))
        loss = c_err + reg
        return np.sum(p_err), c_err, reg, loss

    def optim_user(self, c_mat, p_mat):
        yT = self.item_mat.T
        for u in range(self.n_users):
            Cu = np.diag(c_mat[u])
            yT_Cu_y = yT.dot(Cu).dot(self.item_mat)
            lI = self.alpha * np.identity(self.dim_latent)
            yT_Cu_pu = yT.dot(Cu).dot(p_mat[u])
            self.user_mat[u] = np.linalg.solve(yT_Cu_y + lI, yT_Cu_pu)

    def optim_item(self, c_mat, p_mat):
        xT = self.user_mat.T
        for i in range(self.n_items):
            Ci = np.diag(c_mat[:, i])
            xT_Ci_x = xT.dot(Ci).dot(self.user_mat)
            lI = self.alpha * np.identity(self.dim_latent)
            xT_Ci_pi = xT.dot(Ci).dot(p_mat[:, i])
            self.item_mat[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)


# 학습 준비
# train data + valid(test) data
concat_mat = pd.concat([train, test], axis=0)[['id', 'songs']]
# 장르 정보 + title 형태소 정보 merge
concat_mat = concat_mat.merge(meta[['id', 'song_gn_gnr_basket', 'song_gn_dtl_gnr_basket']], on='id', how='left')
concat_mat = concat_mat.merge(decom_train[['id', 'keywords']], on='id', how='left')

index = concat_mat['id']
columns = [i for i in range(len(enc.classes_) + len(enc_gnr.classes_) + len(enc_dtl_gnr.classes_) + len(enc_title.classes_))]

# rating-matrix 생성(0행)
data = list(enc.transform([concat_mat.iloc[0, 1]]).sum(axis=0))
data += list(enc_gnr.transform([concat_mat.iloc[0, 2]]).sum(axis=0))
data += list(enc_dtl_gnr.transform([concat_mat.iloc[0, 3]]).sum(axis=0))
data += list(enc_title.transform([concat_mat.iloc[0, 4]]).sum(axis=0))
data = [data]
# rating-matrix 생성(1행~)
for _, c in concat_mat[1:].iterrows():
    one_hot = list(enc.transform([c[1]]).sum(axis=0))
    one_hot += list(enc_gnr.transform([c[2]]).sum(axis=0))
    one_hot += list(enc_dtl_gnr.transform([c[3]]).sum(axis=0))
    try:
        one_hot += list(enc_title.transform([c[4]]).sum(axis=0))
    except Exception:
        one_hot += list(enc_title.transform([[]]).sum(axis=0))
    data.append(one_hot)
data = np.array(data)

# matrix factorization 객체 생성
mf = MF(rating_mat=data,
        dim_latent=100,
        l2=0.01,
        alpha=40,
        l_rate=0.01,
        n_epochs=6
        )

pred, p_err, c_err, reg, loss = mf.train()

np.save('test_songs_pred.npy', pred)
print(p_err)
print(c_err)
print(reg)
print(loss)