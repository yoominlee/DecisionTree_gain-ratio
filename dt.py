import sys
import math
import numpy as np

feature_val = []
node = []
feature = []
cc = 0
class_reversed = {}

def read_train_set():
    global feature_val
    global feature
    train_set = open(sys.argv[1], 'r') # open(파일 이름, 열기모드(r/w/a))

    raw_feature = train_set.readline() # 첫줄인 feature 부분만 한줄로
    raw_data = train_set.readlines() # list 형태임.

    feature = raw_feature.replace("\n", "").split('\t')

    # feature_val = [] # 각 feature마다,,
    for i in range(len(feature)):
        feature_val.append([])

    data=[]
    for one_transaction in raw_data:
        # t = one_transaction.replace("\n", "").split()
        t = one_transaction.replace("\n", "").split('\t')
        temp = [] # 전체 데이터인 data의 한 transaction. (한 temp 마다 n-1개의 feature와 class 포함)
        for i in range(len(t)):
            feature_val[i].append(t[i])
            temp.append(t[i])
        data.append(temp)

    # feature_val에 다 append 해뒀던 것 set으로 중복 제거
    for i in range(len(feature_val)):
        feature_val[i] = list(set(feature_val[i]))
        feature_val[i] = {string: j for j, string in enumerate(feature_val[i])} # 딕셔너리로 바꿈


    # print(feature_val)
    # print(feature_val)  #[{'<=30', '>40', '31...40'}, {'high', 'medium', 'low'}, {'yes', 'no'}, {'excellent', 'fair'}, {'yes', 'no'}]
    # [{'<=30': 0, '>40': 1, '31...40': 2}, {'medium': 0, 'high': 1, 'low': 2}, {'yes': 0, 'no': 1},
    #      {'excellent': 0, 'fair': 1}, {'yes': 0, 'no': 1}]
    # 딕셔너리로 바꿈

    # for i in feature_val:
    #     print(i)
    return feature, data

def gain_ratio(feature_num, data): # data, 특정 feature num 넣어주면 그 때의 gain ratio return
    global feature_val
    global class_reversed
    # 분할 기준이 될 feature A를 선택 시
    # 분할 된 각 파티션이 각 항이 되고, 그 안엔 class들 중복해서 들어감

    # partition은 feature A 종류 개수만큼의 길이로 초기화
    partition = [[] for i in range(len(feature_val[feature_num]))]
    for d in data:
        # print(d[feature_num])
        # print(type(feature_val))
        p = feature_val[feature_num]
        # print(p[d[feature_num]])
        partition[p[d[feature_num]]].append(d[-1])

    # print(partition) # [['yes', 'yes', 'no', 'yes', 'no'], ['yes', 'yes', 'yes', 'yes'], ['no', 'no', 'no', 'yes', 'yes']]


    class_reversed = {v: k for k, v in feature_val[-1].items()} # 반복문 안에서 class의 값으로 key를 찾는것 반복해서 순서 바꾼 것 따로 저장
    c = [[] for i in range(len(partition))]
    for p in range(len(partition)):
        unique, counts = np.unique(partition[p], return_counts=True)

        # c[p] = counts # counts 그냥 넣었었는데 0인경우는 생략해서, 분리 전 gain ratio 따로 구해야해서

        # class 개수와 종류에 맞게 저장 위해. yes만 있기도 하고 no만 있기도 함
        # print(len(feature_val[-1])) # 2
        # print(feature_val[-1]) # {'yes': 0, 'no': 1}
        t = dict(zip(unique, counts))
        for feat in range(len(feature_val[-1])): # class 종류만큼 반복. Yes No 인 경우 2회
            if class_reversed[feat] in t: # Yes가 t에 있으면,
                c[p].append(t[class_reversed[feat]])
            else:
                c[p].append(0)

        # print(unique)
        # print(counts)

    gain_ratio_arr = ii(c)
    # print(gain_ratio_arr)
    return gain_ratio_arr


def ii(splits): # i([4,4],[3,0]) 이런식으로 사용?
    # print("-- def ii(splits) --      splits: ", splits)
    total = 0
    each_p = [0 for i in range(len(splits))] # split는 한 특징에 의한 각 파티션인데, each_p에는 각 파티션 별 개수
    each_f = [0 for i in range(len(splits[0]))] # split는 한 특징에 의한 각 파티션인데, each_f에는 각 특징 별 개수. 분리 전 Info 계산 위한 것

    for i in range(len(splits)):
        for j in range(len(splits[i])):
            total += splits[i][j]
            each_p[i] += splits[i][j]
            each_f[j] += splits[i][j]

    # print("each_p: ", each_p)
    # print("each_f: ", each_f)

    # [[0, 4], [2, 3], [3, 2]]
    # each_p:  [4, 5, 5]
    # each_f:  [5, 9]

    info = 0
    info_all = 0
    split_i = 0
    for k in range(len(each_p)):
        if each_p[k] != 0:
            split_i += -(each_p[k]/total)*math.log2(each_p[k]/total)
            for l in range(len(splits[k])):
                u = splits[k][l]
                o = each_p[k]
                # print("u: ", u, " / o: ", o, " / k: ",k, " / total: ", total, " / l: ", l)
                if u!=0: info += (o/total)*(-(u/o)*math.log2(u/o))

                if (k==0) and (each_f[l]!=0): info_all += -(each_f[l]/total)*math.log2(each_f[l]/total)



    # print(info)
    # print(split_i)
    # print(info_all)
    return (info_all - info)/split_i


class Node:
    def __init__(self, data, feature, is_leaf=False, addition=None):
        # parent가 []이면 root로?
        self.feature = feature # 누적 feature
        self.data = data
        # self.parent = parent
        self.child = []
        self.is_leaf = is_leaf
        if len(data)!=0:
            self.class_label = label(self.data)
        else:
            self.class_label = addition
        # print(self.class_label, "       **")
        self.depth = 0

def make_node(data, used_feature):
    global feature_val
    global feature
    # global node
    # if len(used_feature)==0: # root인 경우

    count = []
    for d in data:
        count.append(d[-1])

    unique = np.unique(count) # data  내부의 class들의 종류
    if len(unique) == 1: # class 한종류만 있는 경우.
        # print("9 >> class lable: ", unique)
        return Node(data=data, feature=used_feature, is_leaf=True)

    if used_feature is not None and len(used_feature) == (len(data[0])-1):
        return Node(data=data, feature=used_feature, is_leaf=True)

    select = gain_compare(data, used_feature) # used feature는 사용된 특징의 인덱스 번호로
    # print("1 >> select feature: ", select, " -> ", feature[select])

    # Node(select, data)  # 분리할 때 사용 할 Gain이 가장 큰 특징, 그때의 gain(XX), data, feature 값들
    update_used_feature = used_feature + [select]
    # print("2 >> update used feature: ", update_used_feature)


    # node = Node(data=data,feature=select)

    node = Node(data=data,feature=update_used_feature)
    # print("3 >> create node. {len(data): ", len(data), "}")
    # print("3 >> create node. { len(data): ", len(data), "}")


    count = 0
    node.child = []
    for i in feature_val[select]:
        partition_data = []
        for d in data:
            if d[select]==i: partition_data.append(d)

        if len(partition_data)>=1:
            # nodenum = len(node)
            # print(" + + + ",i)
            # for p in partition_data:
            #     print("     ",i," ",p) # low   ['low', 'vhigh', '5more', '2', 'big', 'low', 'unacc']
                                         # low   ['high', 'high', '5more', '4', 'small', 'low', 'unacc']

            # print("4-1 >> new node. { len(partition_data): ", len(partition_data), "used_feat: ",update_used_feature, "}")
            # print("     000000000")
            # print("feature_val[select]: ",feature_val[select])
            # print("i: ",i)
            # print("feature_val[select][i]: ",feature_val[select][i])
            # print("len(node.child): ",len(node.child))
            node.child.append(make_node(partition_data, update_used_feature))
            # node.child[feature_val[select][i]] = make_node(partition_data, update_used_feature)

        else:
            # print("4-2 >> new leaf node. ")
            # print(partition_data)
            # data가 없는데 node 생성 시 classification 하고싶어해서 오류나는 것을 방지하기 위해
            # 부모의 classlabel을 추가로 넘겨줌.
            # print(node.class_label)
            node.child.append(Node(data=partition_data, feature=used_feature, is_leaf=True, addition=node.class_label))
        count += 1
    return node

def label(data):
    global feature_val
    cla = []
    for d in data:
        cla.append(d[-1])
    # print(cla)
    # print(len(cla))
    unique, counts = np.unique(cla, return_counts=True)
    # t = dict(zip(unique, counts))

    # print(unique)
    # print(counts)
    counts = list(counts)
    # print("--",counts)
    # print("--",unique)
    # print(t)
    tmp = unique[counts.index(max(counts))]

    # print("00000000000000000000000000000")
    # print(feature_val[-1])
    # print(tmp)
    label = feature_val[-1][tmp] # global 변수에 지정되어있는 번호로 return. 첫번째 No Yes data의 경우 Yes이면 1 return
    # print(label)
    return label


def gain_compare(data, used_feature):
    global feature_val
    gain_compare = []
    for i in range(len(feature) - 1):
        if used_feature!=None:
            if i in used_feature: gain_compare.append(-999) # used_feature에 뭔가가 있고, i가 거기에 있을때 (안사용하겠다)
            else: gain_compare.append(gain_ratio(i, data)) # used_feature에 뭔가가 있지만, i가 거기에 없으면 (계산대상)
        else: gain_compare.append(gain_ratio(i, data)) # used_feature에 아무것도 없으면 다 계산 대상

    # if len(used_feature) != 0: # 사용한 feature가 max로 뽑히지 않도록(에러 방지 위해 root인 경우 제외)
    #     for u in used_feature:
    #         gain_compare[u] = 0
    tmp = max(gain_compare)
    select = gain_compare.index(tmp)
    return select

# def makeNode(select, tmp, data): # 분리할 때 사용 할 Gain이 가장 큰 특징, 그때의 gain, data, feature 값들
#
#     global feature_val
#     Node(select, data, )
def read_test():
    test_file = open(sys.argv[2], 'r') # open(파일 이름, 열기모드(r/w/a))

    raw_feature = test_file.readline() # 첫줄인 feature 부분만 한줄로
    raw_data = test_file.readlines() # list 형태임.

    feature = raw_feature.replace("\n", "").split('\t')


    # feature_val = [] # 각 feature마다,,
    # for i in range(len(feature)):
    #     feature_val.append([])

    data=[]
    for one_transaction in raw_data:
        # t = one_transaction.replace("\n", "").split()
        t = one_transaction.replace("\n", "").split('\t')
        temp = [] # 전체 데이터인 data의 한 transaction. (한 temp 마다 n-1개의 feature와 class 포함)
        for i in range(len(t)):
            # feature_val[i].append(t[i])
            temp.append(t[i])
        data.append(temp)

    # # feature_val에 다 append 해뒀던 것 set으로 중복 제거
    # for i in range(len(feature_val)):
    #     feature_val[i] = list(set(feature_val[i]))
    #     feature_val[i] = {string: j for j, string in enumerate(feature_val[i])} # 딕셔너리로 바꿈

    # print(feature)
    # print(data)
    return feature, data, raw_feature, raw_data

def search_node(node, data):
    global feature
    global feature_val
    classify = []
    initial_node = node

    for one_t in data:
        # print(" ====== transaction ",one_t," =====")
        node = initial_node
        feat_str = one_t[node.feature[-1]] # tree 첫번째 구분 feature에 대한 첫 transaction의 값 (node.feature[-1]은, train.txt의 경우 age index)
        feat_int = feature_val[node.feature[-1]][feat_str] # 위에서 찾은 값에 따라 몇번째 child로 갈지

        # print("first transactions 구분 feature: ", feat)        # first transactions 구분 feature:  <=30
        # ff = feature_val[node.feature[-1]]
        # print(ff)                                               # {'<=30': 0, '>40': 1, '31...40': 2}
        # ff = ff[feat]
        # print("그 구분 feature의 num: ", ff)                     # 그 구분 feature의 num:  0

        # 반복문을 통해 leaf로 class label 찾아감
        while not node.is_leaf:
            node = node.child[feat_int]
            # select_feat_vals = feature_val[node.feature[-1]]
            # print("++ node.feature ++ ", node.feature)
            if len(node.feature) != 0:
                # print(node.class_label)
                # feat = data[node.feature[-1]]


                feat_str = one_t[node.feature[-1]] # 해당 노드가 분류되는 데 사용한 feature(node.feature[-1])에 해당하는 현재 선택된 transaction의 값. -> 이거따라 child node 결정
                feat_int = feature_val[node.feature[-1]][feat_str] # 위에서는 해당하는 값(string 형태) 찾은것이고, 여기서는 string에 해당되는 value값.
                # print(select_feat_vals[feat])

                # print("0000000000000000000")
                # print(feature_val[node.feature[-1]]) # {'>40': 0, '31...40': 1, '<=30': 2}
                # print(feat_str) # 31...40
                # print(feat_int) # 1

        # print("CLASS LABEL : ",node.class_label)
        # print(feature_val[-1])
        classify.append(node.class_label)
    return classify

def make_file(classify, raw_feature, raw_data):
    global class_reversed
    output_file = open(sys.argv[3], 'w') # open(파일 이름, 열기모드(r/w/a))
    # print(raw_feature)

    # print(class_reversed) # {0: 'yes', 1: 'no'}  (참고: 원래 형태 {'yes': 0, 'no': 1})

    output_file.write(f'{raw_feature}')


    # print(raw_data)
    for raw in range(len(raw_data)):
        raw_data[raw] = raw_data[raw].replace("\n", "")
        class_lable = class_reversed[classify[raw]]
        output_file.write(f'{raw_data[raw]}\t{class_lable}\n')

    # print(classify)



    output_file.close()

if __name__ == '__main__':
    # 저번 과제는 파일, [1]minsup, [2]input, [3]output
    # 이번 과제는 파일, train.txt, test.txt, result.txt
    # train 각 파일은 맨 윗줄에 특징들 n개. n-1개는 categorical. n번째는 class label
    # test set은 training과 동일한데 class label만 없음.

    feature, data = read_train_set()

    # print(feature)
    # for d in data:
    #     print(d)

    node = make_node(data, [])

    # print("++++++++++++++++++++++++++++++++++++++++++++++")
    feature, data, raw_feature, raw_data = read_test()
    classlabel = search_node(node, data)

    make_file(classlabel, raw_feature, raw_data)


    # print(node.feature)
    # print(node.child[node.feature[-1]])

    print("== FINISH ==")