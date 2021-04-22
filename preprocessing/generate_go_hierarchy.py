import numpy as np
import os
from sys import argv
import requests
script, folder = argv

def gen_direct_ancestor_dic(go_obo_file):
    direct_ancestor_dic = {}
    fr = open(go_obo_file, 'r')
    entry = fr.readline()
    while entry != '':
        if entry == '[Term]\n':
            item = fr.readline().split('\n')[0]
            GOterm = []
            this_go = ''
            while item != '':
                label = item.split(': ')[0]
                if label == 'id':
                    this_go = item.split(': ')[1].split('\n')[0]
                if label == 'is_a':
                    GOterm.append(item.split(': ')[1].split(' ! ')[0])
                if label == 'relationship' or label == 'intersection_of':
                    if item.split(' ')[1].split(' ')[0] == 'part_of':
                        GOterm.append(item.split(' ')[2].split(' ! ')[0])
                item = fr.readline().split('\n')[0]
            direct_ancestor_dic[this_go] = GOterm
        entry = fr.readline()
    fr.close()

    return direct_ancestor_dic

def gen_go_ancestor_dic(direct_ancestor_dic):
    go_ancestor_dic = {}

    for go in direct_ancestor_dic.keys():
        ancestor = []
        ancestor_now = [go]
        while True:
            ancestor_update = list(set(ancestor_now).difference(ancestor))
            if len(ancestor_update) == 0:
                break

            ancestor = list(set(ancestor_now))
            ancestor_now = list(set(ancestor_now))

            for ancestor_go in ancestor_update:
                ancestor_now.extend(direct_ancestor_dic[ancestor_go])

        go_ancestor_dic[go] = list(set(ancestor_now))

    return go_ancestor_dic

if __name__=='__main__':
    print('Generate GO hierarchy...')
    go_obo_file = '../' + folder + '/GO_terms/go.obo'
    if not os.path.exists(go_obo_file):
        url='http://purl.obolibrary.org/obo/go.obo'
        r = requests.get(url, allow_redirects=True)
        open(go_obo_file, 'wb').write(r.content)

    direct_ancestor_dic = gen_direct_ancestor_dic(go_obo_file)
    print(len(direct_ancestor_dic.keys()))

    go_ancestor_dic = gen_go_ancestor_dic(direct_ancestor_dic)
    np.save('../' + folder + '/GO_terms/go_ancestors.npy', np.array([go_ancestor_dic]))
    print(go_ancestor_dic['GO:0005515'])
