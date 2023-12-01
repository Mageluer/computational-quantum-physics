import numpy as np
import numpy.linalg as la
from scipy.sparse import linalg as scilas
from scipy import linalg as scila
import json
import copy


def ncostmnp_route(a):
    d = {}
    ind = list(range(len(a)))

    def cost(ind):
        key = str(a[ind, :][:, ind])
        if key in d:
            return d[key]
        elif len(ind) == 1:
            return 0, str(ind), a[0, 0]
        elif len(ind) == 2:
            if len(a) == 2:
                oprod = 1
            else:
                oind = [i for i in range(len(a)) if i not in ind]
                oprod = np.prod(a[ind, :][:, oind])
            prod = a[ind[0], ind[0]] * a[ind[1], ind[1]]
            mincost = prod * a[ind[0], ind[1]] * oprod
            route = str(ind)
        else:
            if len(ind) == len(a):
                oprod = 1
            else:
                oind = [i for i in range(len(a)) if i not in ind]
                oprod = np.prod(a[ind, :][:, oind])
            prod = a[ind[-1], ind[-1]]*cost(ind[:-1])[2]
            mincost = (prod * np.prod(a[ind[-1], ind[:-1]]) *
                       oprod + cost(ind[:-1])[0])
            route = '[' + cost(ind[:-1])[1] + ', ' + str(ind[-1]) + ']'
            for i in range(len(ind)-2, -1, -1):
                prod = a[ind[i], ind[i]] * cost(ind[:i]+ind[i+1:])[2]
                currcost = (prod *
                            np.prod(a[ind[i], ind[:i]+ind[i+1:]]) *
                            oprod + cost(ind[:i]+ind[i+1:])[0])
                if currcost < mincost:
                    mincost = currcost
                    route = ('[' + cost(ind[:i]+ind[i+1:])[1] +
                             ', ' + str(ind[i]) + ']')
            for i in range(2, len(ind)//2+1):
                ind1_ = list(combinations(ind, i))  # ind1 is shorter than ind2
                ind2_ = [ind[:] for i in range(len(ind1_))]
                for ind1, ind2 in zip(ind1_, ind2_):
                    for i in ind1:
                        ind2.remove(i)
                for ind1, ind2 in zip(ind1_[::-1], ind2_[::-1]):
                    prod = cost(list(ind1))[2] * cost(list(ind2))[2]
                    currcost = (prod *
                                np.prod(a[ind1, :][:, ind2]) * oprod +
                                cost(list(ind1))[0] + cost(list(ind2))[0])
                    if currcost < mincost:
                        mincost = currcost
                        route = ('[' + cost(ind2)[1] + ', ' +
                                 cost(list(ind1))[1] + ']')
        d[key] = (mincost, route, prod)
        return mincost, route, prod
    return cost(ind)


def nfoocost(a):
    def foocost(a, cost=0):
        if len(a) < 2:
            foocost_.append(cost)
            return
        for i, j in combinations(range(len(a)), 2):
            ind = [i for i in range(len(a)) if i != j]
            b = a[:, ind][ind, :]
            b[i, :] = a[i, ind]*a[j, ind]
            b[:, i] = a[ind, i]*a[ind, j]
            b[i, i] = a[i, i]*a[j, j]
            foocost(b, cost+np.prod(a[i, :])*np.prod(a[j, :])//a[i, j])
    foocost_ = []
    foocost(a)
    return min(foocost_)


def route_c2i(route_couple):
    route_ind = list(range(len(route_couple)+1))
    for i, j in route_couple:
        route_ind[i] = [route_ind[i], route_ind[j]]
        route_ind.pop(j)

    return route_ind


def route_i2c(route_ind):
    def ind(a):
        while type(a) == list:
            a = a[0]
        return a

    route_couple = []
    route_ind = [route_ind]

    contain_list = True
    while contain_list:
        for i in range(len(route_ind)-1, -1, -1):
            if type(route_ind[i]) == list:
                route_ind.extend(route_ind[i])
                route_ind.pop(i)
                order = list(np.argsort([ind(i) for i in route_ind]))
                route_couple.insert(
                    0, [order.index(len(route_ind)-2),
                        order.index(len(route_ind)-1)])
                route_ind = [route_ind[i] for i in order]
                break
        else:
            contain_list = False

    return route_couple


def get_adjacent_matrix(tensor_list, leg_links):
    n = len(tensor_list)
    A = np.ones((n, n), dtype=int)
    for i in range(n):
        for k, leg_ind in enumerate(leg_links[i]):
            if leg_ind < 0:
                A[i, i] *= tensor_list[i].shape[k]
        for j in range(i+1, n):
            for k, ind in enumerate(leg_links[i]):
                if ind in leg_links[j]:
                    A[i, j] *= tensor_list[i].shape[k]
            A[j, i] = A[i, j]

    return A


def get_leg_sequence(tensor_list, leg_links, route_couple):
    leg_links = copy.deepcopy(leg_links)
    sequence = []
    for index_tensor, leg_link in enumerate(leg_links):
        leg_trace = []
        leg_dim = []
        for leg in leg_link:
            if leg_link.count(leg) == 2:
                leg_trace.append(leg)
                leg_dim.append(
                    tensor_list[index_tensor].shape(leg_link.index(leg)))
        order = np.argsort(leg_dim)[::-1]
        sequence.extend(np.array(leg_trace)[order])
        leg_links[index_tensor] = set(leg_link)

    for i, j in route_couple:
        sequence.extend(list(leg_links[i] & leg_links[j]))
        leg_links[i] = leg_links[i] ^ leg_links[j]
        leg_links.pop(j)

    return sequence


def get_optimum_sequence(tensor_list, leg_links, mode=0):
    A = get_adjacent_matrix(tensor_list, leg_links)
    route_ind = json.loads((ncostmnp_route(A)[1]))
    if mode == 2:
        return route_ind
    route_couple = route_i2c(route_ind)
    if mode == 1:
        return route_couple
    else:
        return get_leg_sequence(tensor_list, leg_links, route_couple)


def ncon_tlist(tensor_list, leg_links, route_couple):
    leg_links = copy.deepcopy(leg_links)
    for index_tensor, leg_link in enumerate(leg_links):
        axes_trace = []
        leg_dim = []
        for axis1, leg in enumerate(leg_link):
            if leg_link.count(leg) == 2:
                axis2 = leg_link.index(leg, axis1+1)
                axes_trace.append([axis1, axis2])
        order = np.argsort(leg_dim)[::-1]
        for i in order:
            tensor_list[index_tensor] = np.trace(
                tensor_list[index_tensor], *axes_trace[i])
            leg_link.pop(axes_trace[i][1])
            leg_link.pop(axes_trace[i][0])
    for i, j in route_couple:
        common_leg_link = list(set(leg_links[i]) & set(leg_links[j]))
        print(common_leg_link)
        if common_leg_link:
            axis1 = [leg_links[i].index(leg)
                     for leg in common_leg_link]
            axis2 = [leg_links[j].index(leg)
                     for leg in common_leg_link]
            print(axis1, axis2)
            tensor_list[i] = np.tensordot(
                tensor_list[i],
                tensor_list[j],
                (axis1, axis2))
            tensor_list.pop(j)
            for ax1, ax2 in zip(sorted(axis1, reverse=True),
                                sorted(axis2, reverse=True)):
                leg_links[i].pop(ax1)
                leg_links[j].pop(ax2)
            leg_links[i] += leg_links[j]
            leg_links.pop(j)
        else:
            tensor_list[i] = np.tensordot(tensor_list[i], tensor_list[j], 0)
            tensor_list.pop(j)
            leg_links[i] += leg_links[j]
            leg_links.pop(j)
    return tensor_list[0]
