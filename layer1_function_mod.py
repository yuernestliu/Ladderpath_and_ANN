#!/usr/bin/env python
# coding: utf-8



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
import tensorflow.keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
from tensorflow import keras
import tensorflow.compat.v1 as tf1
import random
import ladderpath as lp
import keras_tuner


def load_data(path = r'./mod2dim3.xlsx'):
    data =  pd.read_excel(path)
    data_array = np.array(data)
    X_data = data_array[:,0:3] #X
    y_data = data_array[:,3] #y
    X_data.shape,y_data.shape
    #import random
    np.random.seed(12345)
    permutation = np.random.permutation(y_data.shape[0])
    X_train0 = X_data[permutation, :]
    y_train0 = y_data[permutation]
    X_train = X_train0[0:900, :]
    y_train = y_train0[0:900]
    X_test = X_train0[900:1000, :]
    y_test = y_train0[900:1000]
    #print(y_test)
    return X_train,y_train,X_test,y_test

def change_word(a,eta_random_time=10):
    total_list = []
    word_index = [1,3]
    for i in range(len(a)-1):
        qwe = np.array([1,3])+(i+1)*4+i+1
        word_index = word_index+qwe.tolist() 
    for i in range(len(a)):
        for j in range(len(a[i])):
            total_list.append(a[i][j]) 
    total_array = np.array(total_list) 
    word_array = total_array[word_index]
   # random.seed(seed)
    list_change_total = []
    order_total = []
    for i in range(eta_random_time):#Indicate the number of times to randomly select the minimum value
        
        random.shuffle(word_array)
        total_array[word_index] = word_array 

        list_change = []
        for i in range(len(a)):
            str_name = total_array[5*i]+total_array[5*i+1]+total_array[5*i+2]+total_array[5*i+3]+total_array[5*i+4]#+total_array[7*i+5]+total_array[7*i+6]
            list_change.append(str_name)

        list_change_total.append(list_change)
        
        strs_lp = lp.ladderpath(list_change, CalPOM=False)
        index3 = strs_lp.disp3index()
        strs_lp.dispPOM()
        order = index3[1]
        order_total.append(order)
    min_index =  np.where(order_total==np.min(order_total))[0][0]
    list_change_min = list_change_total[min_index]
    order_min = order_total[min_index]
    return list_change_min,order_min

def max_order(a):
    total_list = []
    word_index = [1,3]
    for i in range(len(a)-1):
        qwe = np.array([1,3])+(i+1)*4+i+1
        word_index = word_index+qwe.tolist() #Get the index of the sequence of Chinese characters
    for i in range(len(a)):
        for j in range(len(a[i])):
            total_list.append(a[i][j]) #Unpack all characters
    total_array = np.array(total_list) 
    word_array = total_array[word_index]
   # random.seed(1)
    #random.shuffle(word_array)
    total_array[word_index] = word_array[0] 
        
    list_change = []
    for i in range(len(a)):
        str_name = total_array[5*i]+total_array[5*i+1]+total_array[5*i+2]+total_array[5*i+3]+total_array[5*i+4]#+total_array[7*i+5]+total_array[7*i+6]
        list_change.append(str_name)
   

    strs_lp = lp.ladderpath(list_change, CalPOM=False)
    index3 = strs_lp.disp3index()
    strs_lp.dispPOM()
    order = index3[1]
    return list_change,order
#Train and save parameters function
def create_model(x,y,period,epochs,min_maxval=[-1,1],lr=0.25,layer123=[8,2],seed=1,seed_1=7,seed_2=10,input_dim=3):#period represents how often to save, epochs is the number of training rounds
    import random
    #random.seed(seed)
   # np.random.seed(seed)
    #tf.random.set_seed(seed)
    model = Sequential()
    model.add(tensorflow.keras.layers.Dense(layer123[0], input_dim=input_dim,kernel_initializer=keras.initializers.RandomUniform(minval=min_maxval[0], maxval=min_maxval[1], seed=seed_1)))#指定下边界和上边界的均匀分布初始化

#,kernel_initializer='ones',bias_initializer='zeros'))
    model.add(tensorflow.keras.layers.Activation('relu'))
    model.add(tensorflow.keras.layers.Dense(layer123[1],kernel_initializer=keras.initializers.RandomUniform(minval=min_maxval[0], maxval=min_maxval[1], seed=seed_2)))
    model.add(tensorflow.keras.layers.Activation('softmax'))
    tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#    model.compile(
#    optimizer=keras.optimizers.Adam(learning_rate=lr),
 #   loss="sparse_categorical_crossentropy",
#    metrics=["accuracy"],)

    checkpoint_path = "./weight_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
 
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=0,save_weights_only=True,period=period)#Save the model's weights every 20 epochs
 
    
    model.save_weights(checkpoint_path.format(epoch=0))
 
    # Train the model using the new callback
    a = model.fit(x,y,epochs=epochs,callbacks=[cp_callback],verbose=0)
    loss_list = a.history['loss']
    # Read the saved model
    layer_name = ['layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE','layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE','layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE',
                  'layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE']

    model_number = epochs/period
    weight = []
    for i in range(int(model_number)):
    
        dic = {}

        checkpoint_path = r'./weight_1/cp-'+str(period*i+period).zfill(4)+'.ckpt' # i.zfill(4)  ensures it is a 4-digit number, e.g., 0001
# Read data from checkpoint file
        reader = tf1.train.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            dic[str(key)] = reader.get_tensor(key)
        weight_change = []
        for i in layer_name:
            weight_change.append(dic[i])
        weight.append(weight_change)
    return weight,loss_list


# # grouping：

def grouping_weight(data,bins=0.1): #Pass the list of weights to be coarse-grained
    weight_copy = copy.deepcopy(data)
    for i in range(len(weight_copy)):   
    
        for j in range(2):
            w = weight_copy[i][j*2]
            for k in range(w.shape[0]):
                for l in range(w.shape[1]):
                    w[k][l] = np.round(w[k][l]/bins)*bins
            #for k in w.flatten():
                #k = np.round(k/bins)*bins
                
    return   weight_copy


#Define the accuracy calculation function
def result_acc(predict_array,true_label):  
    predict_label = []
    for i in predict_array:
        predict_label.append(np.where(i==np.max(i))[0][0])
    predict_right_num = np.sum((np.array(predict_label)==true_label)==1)
    total_num = (np.array(predict_label)==true_label).shape[0]
    acc = predict_right_num/total_num
    
    return acc





# The following function calls grouping_weight to calculate the accuracy of the coarse-grained network after network coarse-graining



def test_bin(weight,acc_X,acc_y,bin_list=[0.1],layer123=[8,2],input_dim=3):
    acc_list = []
    grouping_weight_list = []
    acc_bin_mean_list = []
    #w123 = weight
    for bins in  bin_list:
        weight_copy = copy.deepcopy(weight)
        model_int = Sequential()
        model_int.add(tensorflow.keras.layers.Dense(layer123[0], input_dim=input_dim))#,kernel_initializer='ones',bias_initializer='zeros'))
        model_int.add(tensorflow.keras.layers.Activation('relu'))
        model_int.add(tensorflow.keras.layers.Dense(layer123[1]))
        model_int.add(tensorflow.keras.layers.Activation('softmax'))
        
        weight_copy_grouping = grouping_weight(data=weight_copy,bins=bins) 
        acc_mean_array = np.zeros([len(weight_copy_grouping),1])
        for j in range(len(weight_copy_grouping)):
            
            #model_int.set_weights((np.array(weight_copy_grouping[j])).tolist())#Pass the coarse-grained network into the network to calculate accuracy
            model_int.set_weights((np.array(weight_copy[j])).tolist())#Pass the original network into the network to calculate accuracy
            acc = result_acc(predict_array=model_int.predict(acc_X),true_label=acc_y)
           # acc_mean_array[j] = acc
            
            acc_list.append(acc)
       # acc_mean = np.mean(acc_mean_array)
        
       # acc_bin_mean_list.append(acc_mean)
        grouping_weight_list.append(weight_copy_grouping)
        return acc_list,grouping_weight_list



def print_list_letter_path(weight,ming_unique,bins=0.4):  
    list_letter = []   ## After coarse-graining the networks, label the weights
    for i in range(2):
        list_w = []
        w = weight[i*2].flatten()
        for j in range(len(w)):
            num = int(np.round((w[j]-0)/bins))
            #list_w.append('A_'+str(num))
            index = 500+num 
            list_w.append(ming_unique[index])

        list_w_reshape = np.array(list_w).reshape(weight[i*2].shape[0],weight[i*2].shape[1])# Reshape to the original shape
        list_letter.append(list_w_reshape)
#ming_unique
        
    path_list = [] #After labeling the networks, iterate to obtain all paths
    for i in range(len(list_letter[0])):
        for j in range(len(list_letter[0][i])):
            if list_letter[0][i][j] != ming_unique[500]:
                for k in range(len(list_letter[1][j])):
                    if list_letter[1][j][k] != ming_unique[500]:
                        [list_letter[0][i][j],list_letter[1][j][k]]
#                         for q in range(len(list_letter[2][k])):
#                             if list_letter[2][k][q] != ming_unique[500]:
                        path_list.append([list_letter[0][i][j],list_letter[1][j][k]])
    return list_letter,path_list

#Define a Chinese character dictionary
def load_ming_dic():
    ming=['的', '一', '是', '了', '我', '不', '人', '在', '他', '有', '这', '个', '上', '们', '来', '到', '时', '大', '地', '为',
    '子', '中', '你', '说', '生', '国', '年', '着', '就', '那', '和', '要', '她', '出', '也', '得', '里', '后', '自', '以',
    '会', '家', '可', '下', '而', '过', '天', '去', '能', '对', '小', '多', '然', '于', '心', '学', '么', '之', '都', '好',
    '看', '起', '发', '当', '没', '成', '只', '如', '事', '把', '还', '用', '第', '样', '道', '想', '作', '种', '开', '美',
    '总', '从', '无', '情', '己', '面', '最', '女', '但', '现', '前', '些', '所', '同', '日', '手', '又', '行', '意', '动',
    '方', '期', '它', '头', '经', '长', '儿', '回', '位', '分', '爱', '老', '因', '很', '给', '名', '法', '间', '斯', '知',
    '世', '什', '两', '次', '使', '身', '者', '被', '高', '已', '亲', '其', '进', '此', '话', '常', '与', '活', '正', '感',
    '见', '明', '问', '力', '理', '尔', '点', '文', '几', '定', '本', '公', '特', '做', '外', '孩', '相', '西', '果', '走',
    '将', '月', '十', '实', '向', '声', '车', '全', '信', '重', '三', '机', '工', '物', '气', '每', '并', '别', '真', '打',
    '太', '新', '比', '才', '便', '夫', '再', '书', '部', '水', '像', '眼', '等', '体', '却', '加', '电', '主', '界', '门',
    '利', '海', '受', '听', '表', '德', '少', '克', '代', '员', '许', '稜', '先', '口', '由', '死', '安', '写', '性', '马',
    '光', '白', '或', '住', '难', '望', '教', '命', '花', '结', '乐', '色', '更', '拉', '东', '神', '记', '处', '让', '母',
    '父', '应', '直', '字', '场', '平', '报', '友', '关', '放', '至', '张', '认', '接', '告', '入', '笑', '内', '英', '军',
    '候', '民', '岁', '往', '何', '度', '山', '觉', '路', '带', '万', '男', '边', '风', '解', '叫', '任', '金', '快', '原',
    '吃', '妈', '变', '通', '师', '立', '象', '数', '四', '失', '满', '战', '远', '格', '士', '音', '轻', '目', '条', '呢',
    '病', '始', '达', '深', '完', '今', '提', '求', '清', '王', '化', '空', '业', '思', '切', '怎', '非', '找', '片', '罗',
    '钱', '紶', '吗', '语', '元', '喜', '曾', '离', '飞', '科', '言', '干', '流', '欢', '约', '各', '即', '指', '合', '反',
    '题', '必', '该', '论', '交', '终', '林', '请', '医', '晚', '制', '球', '决', '窢', '传', '画', '保', '读', '运', '及',
    '则', '房', '早', '院', '量', '苦', '火', '布', '品', '近', '坐', '产', '答', '星', '精', '视', '五', '连', '司', '巴',
    '奇', '管', '类', '未', '朋', '且', '婚', '台', '夜', '青', '北', '队', '久', '乎', '越', '观', '落', '尽', '形', '影',
    '红', '爸', '百', '令', '周', '吧', '识', '步', '希', '亚', '术', '留', '市', '半', '热', '送', '兴', '造', '谈', '容',
    '极', '随', '演', '收', '首', '根', '讲', '整', '式', '取', '照', '办', '强', '石', '古', '华', '諣', '拿', '计', '您',
    '装', '似', '足', '双', '妻', '尼', '转', '诉', '米', '称', '丽', '客', '南', '领', '节', '衣', '站', '黑', '刻', '统',
    '断', '福', '城', '故', '历', '惊', '脸', '选', '包', '紧', '争', '另', '建', '维', '绝', '树', '系', '伤', '示', '愿',
    '持', '千', '史', '谁', '准', '联', '妇', '纪', '基', '买', '志', '静', '阿', '诗', '独', '复', '痛', '消', '社', '算',
    '义', '竟', '确', '酒', '需', '单', '治', '卡', '幸', '兰', '念', '举', '仅', '钟', '怕', '共', '毛', '句', '息', '功',
    '官', '待', '究', '跟', '穿', '室', '易', '游', '程', '号', '居', '考', '突', '皮', '哪', '费', '倒', '价', '图', '具',
    '刚', '脑', '永', '歌', '响', '商', '礼', '细', '专', '黄', '块', '脚', '味', '灵', '改', '据', '般', '破', '引', '食',
    '仍', '存', '众', '注', '笔', '甚', '某', '沉', '血', '备', '习', '校', '默', '务', '土', '微', '娘', '须', '试', '怀',
    '料', '调', '广', '蜖', '苏', '显', '赛', '查', '密', '议', '底', '列', '富', '梦', '错', '座', '参', '八', '除', '跑',
    '亮', '假', '印', '设', '线', '温', '虽', '掉', '京', '初', '养', '香', '停', '际', '致', '阳', '纸', '李', '纳', '验',
    '助', '激', '够', '严', '证', '帝', '饭', '忘', '趣', '支', '春', '集', '丈', '木', '研', '班', '普', '导', '顿', '睡',
    '展', '跳', '获', '艺', '六', '波', '察', '群', '皇', '段', '急', '庭', '创', '区', '奥', '器', '谢', '弟', '店', '否',
    '害', '草', '排', '背', '止', '组', '州', '朝', '封', '睛', '板', '角', '况', '曲', '馆', '育', '忙', '质', '河', '续',
    '哥', '呼', '若', '推', '境', '遇', '雨', '标', '姐', '充', '围', '案', '伦', '护', '冷', '警', '贝', '著', '雪', '索',
    '剧', '啊', '船', '险', '烟', '依', '斗', '值', '帮', '汉', '慢', '佛', '肯', '闻', '唱', '沙', '局', '伯', '族', '低',
    '玩', '资', '屋', '击', '速', '顾', '泪', '洲', '团', '圣', '旁', '堂', '兵', '七', '露', '园', '牛', '哭', '旅', '街',
    '劳', '型', '烈', '姑', '陈', '莫', '鱼', '异', '抱', '宝', '权', '鲁', '简', '态', '级', '票', '怪', '寻', '杀', '律',
    '胜', '份', '汽', '右', '洋', '范', '床', '舞', '秘', '午', '登', '楼', '贵', '吸', '责', '例', '追', '较', '职', '属',
    '渐', '左', '录', '丝', '牙', '党', '继', '托', '赶', '章', '智', '冲', '叶', '胡', '吉', '卖', '坚', '喝', '肉', '遗',
    '救', '修', '松', '临', '藏', '担', '戏', '善', '卫', '药', '悲', '敢', '靠', '伊', '村', '戴', '词', '森', '耳', '差',
    '短', '祖', '云', '规', '窗', '散', '迷', '油', '旧', '适', '乡', '架', '恩', '投', '弹', '铁', '博', '雷', '府', '压',
    '超', '负', '勒', '杂', '醒', '洗', '采', '毫', '嘴', '毕', '九', '冰', '既', '状', '乱', '景', '席', '珍', '童', '顶',
    '派', '素', '脱', '农', '疑', '练', '野', '按', '犯', '拍', '征', '坏', '骨', '余', '承', '置', '臓', '彩', '灯', '巨',
    '琴', '免', '环', '姆', '暗', '换', '技', '翻', '束', '增', '忍', '餐', '洛', '塞', '缺', '忆', '判', '欧', '层', '付',
    '阵', '玛', '批', '岛', '项', '狗', '休', '懂', '武', '革', '良', '恶', '恋', '委', '拥', '娜', '妙', '探', '呀', '营',
    '退', '摇', '弄', '桌', '熟', '诺', '宣', '银', '势', '奖', '宫', '忽', '套', '康', '供', '优', '课', '鸟', '喊', '降',
    '夏', '困', '刘', '罪', '亡', '鞋', '健', '模', '败', '伴', '守', '挥', '鲜', '财', '孤', '枪', '禁', '恐', '伙', '杰',
    '迹', '妹', '藸', '遍', '盖', '副', '坦', '牌', '江', '顺', '秋', '萨', '菜', '划', '授', '归', '浪', '听', '凡', '预',
    '奶', '雄', '升', '碃', '编', '典', '袋', '莱', '含', '盛', '济', '蒙', '棋', '端', '腿', '招', '释', '介', '烧', '误',
    '乾', '坤']

    ming_unique = np.unique(np.array(ming))
    return ming_unique

def main_process(weight,layer123,X_train,y_train,X_test,y_test,ming_unique,bin_list=[0.2],input_dim=3):
#     acc_list = []
#     grouping_weight_list = []
    total_path = []
    path_list_total = []
    Ladderpath = []
    Order = []
    
    test_acc_list,grouping_weight_list = test_bin(weight=weight,acc_X=X_test,acc_y=y_test,bin_list=bin_list,layer123=layer123,input_dim=input_dim)
    train_acc_list,grouping_weight_list = test_bin(weight=weight,acc_X=X_train,acc_y=y_train,bin_list=bin_list,layer123=layer123,input_dim=input_dim)
    
    for  i in range(len(bin_list)):#'i' represents the number of coarse-graining interval
        for j in range(len(weight)):#'j' ranges from 0 to 100, representing 100 networks
            list_letter,path_list = print_list_letter_path(weight=weight[j],ming_unique=ming_unique,bins=bin_list[i])
        
            total_path.append(path_list)#Obtain all paths without deduplication, similarly
        
        #path_node_num = complexity_define(path_list)
        #dic[str(i)+str(j)] = path_node_num

    number_list = ['a','b','c','d','e','f']
    #path_list_total = []
    for i in range(len(total_path)):  
    #a=total_path[i]
        path_list_i = []
        for j in range(len(total_path[i])):
    
            apath=''
            for k in range(2):
        
                apath =apath+ total_path[i][j][k]+number_list[k]
            path_list_i.append('d'+apath)
        path_list_total.append(path_list_i)
        

    for i in range(len(path_list_total)):
        strs_lp = lp.ladderpath(path_list_total[i], CalPOM=False)
        index3 = strs_lp.disp3index()
        strs_lp.dispPOM()
        Ladderpath.append(index3[0])
        Order.append(index3[1])
    
    return train_acc_list,test_acc_list,grouping_weight_list,total_path,path_list_total,Ladderpath,Order
#total_path = []
#path_list_total = []

#bin_list = [0.2]
#layer123=[10,12,2]
#X_train,y_train,X_test,y_test = load_data()
#ming_unique = load_ming_dic()

#epochs_weight_list = create_model(X_train,y_train,period=50,epochs=200,lr=0.5,layer123=[10,12,2],seed=3,seed_1=7,seed_2=10,seed_3=1)
