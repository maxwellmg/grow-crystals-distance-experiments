import os
import sys
import re
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import random
from tqdm import tqdm
from itertools import product

import time
import datetime

import csv
import h5py
import multiprocessing

#from transformer_lens import *

from transformers import pipeline

from sklearn.decomposition import PCA
from sklearn.utils import shuffle

def GenerateFamilyTree(nodes_MAX = 200, max_child_per_gen = 6, seed = 33, real_world_data = None):
    dict_level = dict()
    dict_father = dict()
    dict_mother = dict()
    dict_gender = dict()
    dict_spouse = dict()
    dict_sons = dict()
    dict_daughters = dict()
    dict_brothers = dict()
    dict_sisters = dict()
    dict_siblings = dict()

    dict_child = dict()


    dict_couple_idx = dict()
    dict_siblings_idx = dict()
    couple_idx = 0
    siblings_idx = 0

    random.seed(seed)
    np.random.seed(seed)

    if real_world_data == None:
        cur_idx = 0
        bfs_lst = []

        while cur_idx < nodes_MAX: # bfs_lst
        #  print(cur_idx)
            if not bfs_lst:
                print(cur_idx)
                dict_gender[cur_idx] = np.random.randint(2)
                dict_level[cur_idx] = cur_idx
                bfs_lst.append(cur_idx)
                cur_idx += 1

            cur_person = bfs_lst.pop()
            rng_num = np.random.randint(2)

            if rng_num >= 0: # marry with 70% probability
                dict_couple_idx[cur_person] = couple_idx
                dict_couple_idx[cur_idx] = couple_idx
                couple_idx += 1

                dict_spouse[cur_person] = cur_idx
                dict_spouse[cur_idx] = cur_person
                dict_gender[cur_idx] = 1 - dict_gender[cur_person]
                dict_level[cur_idx] = dict_level[cur_person]
                cur_idx += 1
                rng_num = np.random.randint(2)

                if rng_num >= 0: # have children with 70% probability
                    num_child = np.random.randint(1,max_child_per_gen+1)
                    list_sons = []
                    list_daughters = []
                    for j in range(num_child):
                        rng_gen = np.random.randint(2) # boy or girl, each with 50% probability
                        dict_siblings_idx[cur_idx] = siblings_idx
                        if rng_gen == 1:
                            list_sons.append(cur_idx)
                            dict_gender[cur_idx] = 1
                            bfs_lst.append(cur_idx)
                            dict_level[cur_idx] = dict_level[cur_person] + 1
                            cur_idx += 1
                        else:
                            list_daughters.append(cur_idx)
                            dict_gender[cur_idx] = 0
                            bfs_lst.append(cur_idx)
                            dict_level[cur_idx] = dict_level[cur_person] + 1
                            cur_idx += 1
                    siblings_idx += 1

                    dict_sons[cur_person] = list_sons
                    dict_sons[dict_spouse[cur_person]] = list_sons

                    dict_daughters[cur_person] = list_daughters
                    dict_daughters[dict_spouse[cur_person]] = list_daughters

                    for son in list_sons:
                        dict_father[son] = cur_person if dict_gender[cur_person] == 1 else dict_spouse[cur_person]
                        dict_mother[son] = cur_person if dict_gender[cur_person] == 0 else dict_spouse[cur_person]
                        dict_brothers[son] = [item for item in list_sons if item != son]
                        dict_sisters[son] = list_daughters

                    for daughter in list_daughters:
                        dict_father[daughter] = cur_person if dict_gender[cur_person] == 1 else dict_spouse[cur_person]
                        dict_mother[daughter] = cur_person if dict_gender[cur_person] == 0 else dict_spouse[cur_person]
                        dict_brothers[daughter] = list_sons
                        dict_sisters[daughter] = [item for item in list_daughters if item != daughter]
    else:
        girl_names_list = ["Rory", "Virginia", "Tatiana", "Patricia", "Mary","Sydney","Caroline", "Kara", "Amy", "Maria", "Bridget", "Eunice", "Rose", "Ethel", "Joanna", "Victoria", "Robin", "Arabella","Kym", "Kathleen", "Katherine", "Christina", "Margaret", "Jean", "Jacqueline", "Amanda"]
        girl_names_list += ["Swarup", "Vijaya", "Krishna", "Kamala", "Indira", "Rameshwari", "Uma", "Magdolna", "Sarup","Chandralekha", "Nayantara", "Rita", "Shyam", "Sonia", "Maneka", "Subhadra", "Neena", "Samhita", "Priyanka", "Yamini", "Avantika", "Radhika", "Anasuya", "Miraya"]    
        girl_names_list += ["Guttle", "Sch\u00f6nche", "Isabella", "Babette", "Julie", "Henriette", "Eva", "Caroline", "Betty", "Charlotte", "Mathilde", "Alice", "Evelina", "Bettina", "Hannah", "Louise", "Adelheid", "Ad\u00e8le", "Emma", "Clementine", "Laura", "Margarethe", "Bertha", "Leonora"]

        for (x,y,z) in real_world_data:
            if x.split(" ")[0] in girl_names_list:
                dict_gender[x] = 0
            else:
                dict_gender[x] = 1
            if z.split(" ")[0] in girl_names_list:
                dict_gender[z] = 0
            else:
                dict_gender[z] = 1

        for (x,y,z) in real_world_data:
            if y == "father of":
                dict_father[z] = x
            if y == "mother of":
                dict_mother[z] = x
            if y == "married to":
                dict_spouse[x] = z
                dict_spouse[z] = x

        for (x,y,z) in real_world_data:   
            if y == "father of" or y == "mother of":
                if dict_gender[z] == 1:
                    if x not in dict_sons.keys():
                        dict_sons[x] = [z]
                    else:
                        dict_sons[x].append(z)
                    if x not in dict_child.keys():
                        dict_child[x] = [z]
                    else:
                        dict_child[x].append(z)
                else:
                    if x not in dict_daughters.keys():
                        dict_daughters[x] = [z]
                    else:
                        dict_daughters[x].append(z)
                    if x not in dict_child.keys():
                        dict_child[x] = [z]
                    else:
                        dict_child[x].append(z)

        cur_siblings_idx = 1
        for key in dict_child.keys():
            if dict_gender[key] == 1 or (dict_gender[key] == 0 and key not in dict_spouse.keys()):
                sibling_list = dict_child[key]
                for i in range(len(sibling_list)):
                    dict_siblings_idx[sibling_list[i]] = cur_siblings_idx
                cur_siblings_idx += 1

                for i in range(len(sibling_list)):
                    for j in range(len(sibling_list)):
                        if i == j:
                            continue
                        if dict_gender[sibling_list[j]] == 0:
                            if sibling_list[i] not in dict_sisters.keys():
                                dict_sisters[sibling_list[i]] = [sibling_list[j]]
                            else:
                                dict_sisters[sibling_list[i]].append(sibling_list[j])
                        else:
                            if sibling_list[i] not in dict_brothers.keys():
                                dict_brothers[sibling_list[i]] = [sibling_list[j]]
                            else:
                                dict_brothers[sibling_list[i]].append(sibling_list[j])

        cur_couple_idx = 1
        for key in dict_spouse.keys():
            if key not in dict_couple_idx.keys():
                dict_couple_idx[key] = cur_couple_idx
                dict_couple_idx[dict_spouse[key]] = cur_couple_idx
                cur_couple_idx += 1


    col_subject = []; col_relation = []; col_object = []

    for key in dict_father:
        col_subject.append(dict_father[key])
        col_relation.append("Father")
        col_object.append(key)

    for key in dict_mother:
        col_subject.append(dict_mother[key])
        col_relation.append("Mother")
        col_object.append(key)

    for key in dict_spouse:
        if dict_gender[key] == 1:
            col_subject.append(key)
            col_relation.append("Husband")
            col_object.append(dict_spouse[key])
        else:
            col_subject.append(key)
            col_relation.append("Wife")
            col_object.append(dict_spouse[key])
        '''
        col_subject.append(key)
        col_relation.append("Spouse")
        col_object.append(dict_spouse[key])
        '''

    for key in dict_sons:
        tmp_arr = dict_sons[key]
        for son in tmp_arr:
            col_subject.append(son)
            col_relation.append("Son")
            col_object.append(key)

    for key in dict_daughters:
        tmp_arr = dict_daughters[key]
        for daughter in tmp_arr:
            col_subject.append(daughter)
            col_relation.append("Daughter")
            col_object.append(key)

    for key in dict_brothers:
        for brother in dict_brothers[key]:
            col_subject.append(brother)
            col_relation.append("Brother")
            col_object.append(key)

    for key in dict_sisters:
        for sister in dict_sisters[key]:
            col_subject.append(sister)
            col_relation.append("Sister")
            col_object.append(key)

    for key in dict_father:
        if dict_father[key] in dict_father:
            col_subject.append(dict_father[dict_father[key]])
            col_relation.append("Grandfather")
            col_object.append(key)
        if dict_father[key] in dict_mother:
            col_subject.append(dict_mother[dict_father[key]])
            col_relation.append("Grandmother")
            col_object.append(key)

    for key in dict_mother:
        if dict_mother[key] in dict_father:
            col_subject.append(dict_father[dict_mother[key]])
            col_relation.append("Grandfather")
            col_object.append(key)
        if dict_mother[key] in dict_mother:
            col_subject.append(dict_mother[dict_mother[key]])
            col_relation.append("Grandmother")
            col_object.append(key)
    
    for key in dict_father:
        if dict_father[key] in dict_sisters:
            for aunt in dict_sisters[dict_father[key]]:
                col_subject.append(aunt)
                col_relation.append("Aunt")
                col_object.append(key)

                col_subject.append(key)
                col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                col_object.append(aunt)

                if aunt in dict_spouse:
                    col_subject.append(dict_spouse[aunt])
                    col_relation.append("Uncle")
                    col_object.append(key)

                    col_subject.append(key)
                    col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                    col_object.append(dict_spouse[aunt])

        if dict_father[key] in dict_brothers:
            for uncle in dict_brothers[dict_father[key]]:
                col_subject.append(uncle)
                col_relation.append("Uncle")
                col_object.append(key)

                col_subject.append(key)
                col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                col_object.append(uncle)

                if uncle in dict_spouse:
                    col_subject.append(dict_spouse[uncle])
                    col_relation.append("Aunt")
                    col_object.append(key)

                    col_subject.append(key)
                    col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                    col_object.append(dict_spouse[uncle])
                    
    for key in dict_mother:
        if dict_mother[key] in dict_sisters:
            for aunt in dict_sisters[dict_mother[key]]:
                col_subject.append(aunt)
                col_relation.append("Aunt")
                col_object.append(key)

                col_subject.append(key)
                col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                col_object.append(aunt)

                if aunt in dict_spouse:
                    col_subject.append(dict_spouse[aunt])
                    col_relation.append("Uncle")
                    col_object.append(key)

                    col_subject.append(key)
                    col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                    col_object.append(dict_spouse[aunt])

        if dict_mother[key] in dict_brothers:
            for uncle in dict_brothers[dict_mother[key]]:
                col_subject.append(uncle)
                col_relation.append("Uncle")
                col_object.append(key)

                col_subject.append(key)
                col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                col_object.append(uncle)

                if uncle in dict_spouse:
                    col_subject.append(dict_spouse[uncle])
                    col_relation.append("Aunt")
                    col_object.append(key)

                    col_subject.append(key)
                    col_relation.append("Nephew" if dict_gender[key] == 1 else "Niece")
                    col_object.append(dict_spouse[uncle])

    for key in dict_spouse:
        if dict_spouse[key] in dict_brothers:
            for brolaw in  dict_brothers[dict_spouse[key]]:
                col_subject.append(brolaw)
                col_relation.append("Brother-in-law")
                col_object.append(key)

        if dict_spouse[key] in dict_sisters:
            for sislaw in  dict_sisters[dict_spouse[key]]:
                col_subject.append(sislaw)
                col_relation.append("Sister-in-law")
                col_object.append(key)

    for key in dict_sisters:
        for sister in dict_sisters[key]:
            if sister in dict_spouse:
                brolaw = dict_spouse[sister]
                col_subject.append(brolaw)
                col_relation.append("Brother-in-law")
                col_object.append(key)

    for key in dict_brothers:
        for brother in dict_brothers[key]:
            if brother in dict_spouse:
                sislaw = dict_spouse[brother]
                col_subject.append(sislaw)
                col_relation.append("Sister-in-law")
                col_object.append(key)
    
    
    
    for key in dict_gender.keys():
        col_subject.append(key)
        col_relation.append("Descendant")
        col_object.append(key)

        col_subject.append(key)
        col_relation.append("Ancestor")
        col_object.append(key)

        cur_lst = [key]
        while cur_lst:
            cur_idx = cur_lst.pop()
            if cur_idx in dict_father:
                col_subject.append(key)
                col_relation.append("Descendant")
                col_object.append(dict_father[cur_idx])

                col_subject.append(dict_father[cur_idx])
                col_relation.append("Ancestor")
                col_object.append(key)

                cur_lst.append(dict_father[cur_idx])

            if cur_idx in dict_mother:
                col_subject.append(key)
                col_relation.append("Descendant")
                col_object.append(dict_mother[cur_idx])

                col_subject.append(dict_mother[cur_idx])
                col_relation.append("Ancestor")
                col_object.append(key)

                cur_lst.append(dict_mother[cur_idx])
    
    df = pd.DataFrame({'Subject': col_subject, 'Relation': col_relation, 'Object': col_object})
    df.drop_duplicates(inplace=True)
    print(df.info())

    for key in dict_brothers.keys():
        if key not in dict_siblings.keys():
            dict_siblings[key] = dict_brothers[key]
        else:
            dict_siblings[key] += dict_brothers[key]

    for key in dict_sisters.keys():
        if key not in dict_siblings.keys():
            dict_siblings[key] = dict_sisters[key]
        else:
            dict_siblings[key] += dict_sisters[key]

    ret_dic = {}
    ret_dic["nodes_MAX"] = nodes_MAX
    ret_dic["max_child_per_gen"] = max_child_per_gen
    ret_dic["seed"] = seed

    ret_dic["family_data"] = df

    ret_dic["col_subject"] = col_subject
    ret_dic["col_relation"] = col_relation
    ret_dic["col_object"] = col_object

    ret_dic["dict_level"] = dict_level
    ret_dic["dict_father"] = dict_father
    ret_dic["dict_mother"] = dict_mother
    ret_dic["dict_gender"] = dict_gender


    ret_dic["dict_spouse"] = dict_spouse
    ret_dic["dict_sons"] = dict_sons
    ret_dic["dict_daughters"] = dict_daughters
    ret_dic["dict_brothers"] = dict_brothers
    ret_dic["dict_sisters"] = dict_sisters
    ret_dic["dict_child"] = dict_child
    ret_dic["dict_siblings"] = dict_siblings

    ret_dic["dict_couple_idx"] = dict_couple_idx
    ret_dic["dict_siblings_idx"] = dict_siblings_idx

    return ret_dic


def check_father(sub, obj, ret_dic):
    dict_father = ret_dic["dict_father"]
    if obj in dict_father.keys() and dict_father[obj] == sub:
        return True
    return False

def check_mother(sub, obj, ret_dic):
    dict_mother = ret_dic["dict_mother"]
    if obj in dict_mother.keys() and dict_mother[obj] == sub:
        return True
    return False

def check_parent(sub, obj, ret_dic):
    if check_father(sub, obj, ret_dic) or check_mother(sub, obj, ret_dic):
        return True
    return False

def check_spouse(sub, obj, ret_dic):
    dict_spouse = ret_dic["dict_spouse"]
    if obj in dict_spouse.keys() and dict_spouse[obj] == sub:
        return True
    return False

def check_husband(sub, obj, ret_dic):
    dict_spouse = ret_dic["dict_spouse"]
    dict_gender = ret_dic["dict_gender"]
    if obj in dict_spouse.keys() and dict_spouse[obj] == sub and dict_gender[sub] == 1:
        return True
    return False

def check_wife(sub, obj, ret_dic):
    dict_spouse = ret_dic["dict_spouse"]
    dict_gender = ret_dic["dict_gender"]
    if obj in dict_spouse.keys() and dict_spouse[obj] == sub and dict_gender[sub] == 0:
        return True
    return False

def check_son(sub, obj, ret_dic):
    dict_sons = ret_dic["dict_sons"]
    if obj in dict_sons.keys() and sub in dict_sons[obj]:
        return True
    return False

def check_daughter(sub, obj, ret_dic):
    dict_daugthers = ret_dic["dict_daughters"]
    if obj in dict_daugthers.keys() and sub in dict_daugthers[obj]:
        return True
    return False

def check_brother(sub, obj, ret_dic):
    dict_brothers = ret_dic["dict_brothers"]
    if obj in dict_brothers.keys() and sub in dict_brothers[obj]:
        return True
    return False

def check_sister(sub, obj, ret_dic):
    dict_sisters = ret_dic["dict_sisters"]
    if obj in dict_sisters.keys() and sub in dict_sisters[obj]:
        return True
    return False

def check_sibling(sub, obj, ret_dic):
    return check_brother(sub,obj,ret_dic) or check_sister(sub, obj,ret_dic)

def check_grandfather(sub, obj, ret_dic):
    nodes_MAX = ret_dic["nodes_MAX"]
    dict_level = ret_dic["dict_level"]
    dict_gender = ret_dic["dict_gender"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and dict_gender[sub] == 1 and check_parent(sub, j, ret_dic) and check_parent(j, obj, ret_dic):
            return True
    return False

def check_grandmother(sub, obj, ret_dic):
    nodes_MAX = ret_dic["nodes_MAX"]
    dict_level = ret_dic["dict_level"]
    dict_gender = ret_dic["dict_gender"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and dict_gender[sub] == 0 and check_parent(sub, j, ret_dic) and check_parent(j, obj, ret_dic):
            return True
    return False

def check_aunt(sub, obj, ret_dic):
    dict_spouse = ret_dic["dict_spouse"]
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    sub_spo = dict_spouse[sub] if sub in dict_spouse.keys() else -1
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_parent(j, obj, ret_dic) and check_sister(sub, j, ret_dic)) or (sub_spo != -1 and check_uncle_b(sub_spo, obj, ret_dic))):
            return True
    return False

def check_aunt_b(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_parent(j, obj, ret_dic) and check_sister(sub, j, ret_dic))):
            return True
    return False

def check_uncle(sub, obj, ret_dic):
    dict_spouse = ret_dic["dict_spouse"]
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    sub_spo = dict_spouse[sub] if sub in dict_spouse.keys() else -1
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_parent(j, obj, ret_dic) and check_brother(sub, j, ret_dic)) or (sub_spo != -1 and check_aunt_b(sub_spo, obj, ret_dic))):
            return True
    return False

def check_uncle_b(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_parent(j, obj, ret_dic) and check_brother(sub, j, ret_dic))):
            return True
    return False


def check_niece(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and check_daughter(sub, j, ret_dic) and (check_brother(j, obj, ret_dic) or check_sister(j,obj,ret_dic) or check_broinlaw(j,obj,ret_dic) or check_sisinlaw(j,obj,ret_dic)):
            return True
    return False

def check_nephew(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and check_son(sub, j, ret_dic) and (check_brother(j, obj, ret_dic) or check_sister(j,obj,ret_dic) or check_broinlaw(j,obj,ret_dic) or check_sisinlaw(j,obj,ret_dic)):
            return True
    return False

def check_broinlaw(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_brother(sub, j, ret_dic) and check_spouse(j,obj, ret_dic)) or (check_husband(sub, j, ret_dic) and check_sibling(j,obj, ret_dic))):
            return True
    return False

def check_sisinlaw(sub, obj, ret_dic):
    dict_level = ret_dic["dict_level"]
    nodes_MAX = ret_dic["nodes_MAX"]
    for j in range(nodes_MAX):
        if (j in dict_level.keys()) and ((check_sister(sub, j,ret_dic) and check_spouse(j,obj,ret_dic)) or (check_wife(sub, j, ret_dic) and check_sibling(j,obj,ret_dic))):
            return True
    return False


def check_family_relation(sub, rel, obj, ret_dic):
    if rel == "Father":
        return check_father(sub, obj, ret_dic)
    if rel == "Mother":
        return check_mother(sub, obj, ret_dic)
    if rel == "Parent":
        return check_parent(sub, obj, ret_dic)
    if rel == "Spouse":
        return check_spouse(sub, obj, ret_dic)
    if rel == "Husband":
        return check_husband(sub, obj, ret_dic)
    if rel == "Wife":
        return check_wife(sub, obj, ret_dic)
    if rel == "Son":
        return check_son(sub, obj, ret_dic)
    if rel == "Daughter":
        return check_daughter(sub, obj, ret_dic)
    if rel == "Brother":
        return check_brother(sub, obj, ret_dic)
    if rel == "Sister":
        return check_sister(sub, obj, ret_dic)
    if rel == "Sibling":
        return check_sibling(sub, obj, ret_dic)
    if rel == "Grandfather":
        return check_grandfather(sub, obj, ret_dic)
    if rel == "Grandmother":
        return check_grandmother(sub, obj, ret_dic)
    if rel == "Aunt":
        return check_aunt(sub, obj, ret_dic)
    if rel == "Uncle":
        return check_uncle(sub, obj, ret_dic)
    if rel == "Niece":
        return check_niece(sub, obj, ret_dic)
    if rel == "Nephew":
        return check_nephew(sub, obj, ret_dic)
    if rel == "Brother-in-law":
        return check_broinlaw(sub, obj, ret_dic)
    if rel == "Sister-in-law":
        return check_sisinlaw(sub, obj, ret_dic)
    return False