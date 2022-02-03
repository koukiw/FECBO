import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sysはコマンドライン引数
import sys
import csv
np.seterr(divide='ignore', invalid='ignore')
#csvファイルの読み込み、一行目はヘッダーじゃない
df = pd.read_csv("output.csv",header=None)
#クラスタ数宣言
c=2
#Kはクラスタサイズ
K=5
#dfをnumpyで扱えるnp.ndarray変換
X = df.values
#配列の行数、列数の格納
X_size,dimension = X.shape
centroids = X[np.random.choice(X_size,c)]
distances=np.zeros((X_size,c))

#dにはd11,d12,d13,d14,d21,d22,,,dn1,,dncの順でリスト作る
d=[]
for i in range(X_size):
    for j in range(c):
        d.append(np.sum((X[i]-centroids[j])**2))
d = [0.0000000001 if i == 0 else i for i in d]

from sympy import diff,symbols,solve

#帰属度uの変数定義
u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19 = symbols("u0 u1 u2 u3 u4 u5 u6 u7 u8 u9 u10 u11 u12 u13 u14 u15 u16 u17 u18 u19")
#条件式の変数定義
g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11 = symbols("g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11")
#ラムダ変数の定義
l0,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11 = symbols("l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11")
#目的関数定義
f = u0**3*d[0] + u1**3*d[1] + u2**3*d[2] + u3**3*d[3] + u4**3*d[4] + u5**3*d[5] +u6**3*d[6] + u7**3*d[7] + u8**3*d[8] + u9**3*d[9] + u10**3*d[10] + u11**3*d[11] +u12**3*d[12] + u13**3*d[13] + u14**3*d[14] + u15**3*d[15] + u16**3*d[16] + u17**3*d[17] +u18**3*d[18]+ u19**3*d[19]
#帰属度が1になる制約式
g0 = u0 + u1 -1
g1 = u2 + u3 -1
g2 = u4 + u5 -1
g3 = u6 + u7 -1
g4 = u8 + u9 -1
g5 = u10 + u11 -1
g6 = u12 + u13 -1
g7 = u14 + u15 -1
g8 = u16 + u17 -1
g9 = u18 + u19 -1
#クラスタサイズがcになる制約式
g10 = u0+u2+u4+u6+u8+u10+u12+u14+u16+u18-5
g11 = u1+u3+u5+u7+u9+u11+u13+u15+u17+u19-5

# 定理式に代入
theor = f-l0*g0-l1*g1-l2*g2-l3*g3-l4*g4-l5*g5-l6*g6-l7*g7-l8*g8-l9*g9-l10*g10-l11*g11
# 各変数で偏微分する
diff_u0 = diff(theor, u0)
diff_u1 = diff(theor, u1)
diff_u2 = diff(theor, u2)
diff_u3 = diff(theor, u3)
diff_u4 = diff(theor, u4)
diff_u5 = diff(theor, u5)
diff_u6 = diff(theor, u6)
diff_u7 = diff(theor, u7)
diff_u8 = diff(theor, u8)
diff_u9 = diff(theor, u9)
diff_u10 = diff(theor, u10)
diff_u11 = diff(theor, u11)
diff_u12 = diff(theor, u12)
diff_u13 = diff(theor, u13)
diff_u14 = diff(theor, u14)
diff_u15 = diff(theor, u15)
diff_u16 = diff(theor, u16)
diff_u17 = diff(theor, u17)
diff_u18 = diff(theor, u18)
diff_u19 = diff(theor, u19)

diff_l0 = diff(theor, l0)
diff_l1 = diff(theor, l1)
diff_l2 = diff(theor, l2)
diff_l3 = diff(theor, l3)
diff_l4 = diff(theor, l4)
diff_l5 = diff(theor, l5)
diff_l6 = diff(theor, l6)
diff_l7 = diff(theor, l7)
diff_l8 = diff(theor, l8)
diff_l9 = diff(theor, l9)
diff_l10 = diff(theor, l10)
diff_l11 = diff(theor, l11)



res = solve([diff_u0, diff_u1, diff_u2, diff_u3, diff_u4, diff_u5, diff_u6, diff_u7, diff_u8, diff_u9, diff_u10, diff_u11,
            diff_u12, diff_u13, diff_u14, diff_u15, diff_u16, diff_u17, diff_u17, diff_u18, diff_u19, diff_l0, diff_l1, diff_l2,
            diff_l3, diff_l4, diff_l5, diff_l6, diff_l7, diff_l8, diff_l9, diff_l10, diff_l11])
print(res)

