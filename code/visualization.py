import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "mathtext.fontset":'stix',
    "font.family":'Times New Roman',
    "font.size": 28,
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False # 处理负号，即-号
}

rcParams.update(config)


y_1=[0.0458,0.0652,0.0779,0.0587]
x_1=[1,2,3,4]
p1 = plt.figure(figsize=(24,10) ,dpi=100)  #画布 长宽8:6,分辨率=80
ax1 = p1.add_subplot(1,2,1)
plt.title('HitRate@10')
plt.xlabel("Number of Layer (L)")
plt.ylabel('HitRate@10')
plt.xlim((1, 4))
plt.ylim((0.04, 0.08))
plt.xticks([1,2,3,4]) #设置x轴刻度
plt.yticks([0.04,0.05,0.06,0.07,0.08]) #设置y轴刻度
plt.plot(x_1, y_1)

y_2=[0.0877,0.0982,0.1146,0.1083]
x_2=[1,2,3,4]
ax2 = p1.add_subplot(1,2,2)
plt.title('NDCG@10')
plt.xlabel("Number of Layer (L)")
plt.ylabel('NDCG@10')
plt.xlim((1, 4))
plt.ylim((0.08, 0.12))
plt.xticks([1,2,3,4]) #设置x轴刻度
plt.yticks([0.08,0.09,0.10,0.11,0.12]) #设置y轴刻度
plt.plot(x_2, y_2)

plt.savefig('C:/Users/KevinXu/Desktop/Layer.png')
plt.show()
'''

y_1=[0.0498,0.0622,0.0779,0.0612,0.0677]
x_1=[1,2,3,4,5]
p1 = plt.figure(figsize=(14,6) ,dpi=80)  #画布 长宽8:6,分辨率=80
ax1 = p1.add_subplot(1,2,1)
plt.title('HitRate@10')
plt.xlabel("Number of Preference (K)")
plt.ylabel('HitRate@10')
plt.xlim((1, 5))
plt.ylim((0.04, 0.08))
plt.xticks([1,2,3,4,5]) #设置x轴刻度
plt.yticks([0.04,0.05,0.06,0.07,0.08]) #设置y轴刻度
plt.plot(x_1, y_1)

y_2=[0.0877,0.1022,0.1146,0.1083,0.1055]
x_2=[1,2,3,4,5]
ax2 = p1.add_subplot(1,2,2)
plt.title('NDCG@10')
plt.xlabel("Number of Preference (K)")
plt.ylabel('NDCG@10')
plt.xlim((1, 5))
plt.ylim((0.08, 0.12))
plt.xticks([1,2,3,4,5]) #设置x轴刻度
plt.yticks([0.08,0.09,0.10,0.11,0.12]) #设置y轴刻度
plt.plot(x_2, y_2)

plt.savefig('C:/Users/KevinXu/Desktop/K.png')
plt.show()
'''