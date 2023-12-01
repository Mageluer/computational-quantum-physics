import numpy as np
import opt_einsum as oe
import copy


def route2sequence(tensor_list, leg_links, route_couple):
    leg_links = copy.deepcopy(leg_links)
    sequence = []
    for index_tensor, leg_link in enumerate(leg_links):
        leg_trace = [leg for leg in leg_link if leg_link.count(leg) == 2]
        leg_dim = [
            tensor_list[index_tensor].shape(leg_link.index(leg)) for leg in leg_trace
        ]
        sequence.extend(np.array(leg_trace)[np.argsort(leg_dim)[::-1]])
        leg_links[index_tensor] = set(leg_link)

    for i, j in route_couple:
        sequence.extend(list(leg_links[i] & leg_links[j]))
        leg_links.append(leg_links[i] ^ leg_links[j])
        leg_links.pop(j)
        leg_links.pop(i)

    return sequence


def get_opt_sequence(
    tensor_list, leg_links, optimize="optimal", memory_limit=None, verbosity=True
):
    leg_str_ = [
        "".join([oe.get_symbol(leg) for leg in leg_link]) for leg_link in leg_links
    ]
    subscript = ",".join(leg_str_) + "->"

    route_couple, path_info = oe.contract_path(
        subscript, *tensor_list, memory_limit=memory_limit
    )

    if verbosity:
        print("Optimal contraction route in tensor couples:", route_couple)
        print(path_info)

    sequence = route2sequence(tensor_list, leg_links, route_couple)

    return sequence


if __name__ == "__main__":
    leg_links = [
        [1, 2, 3, 4],
        [1, 11, 5],
        [5, 12, 6],
        [6, 13, 7],
        [7, 8, 9, 10],
        [2, 11, 14, 20],
        [14, 12, 15, 21],
        [15, 13, 8, 22],
        [3, 20, 16, 23],
        [16, 21, 17, 24],
        [17, 22, 9, 25],
        [4, 23, 18],
        [18, 24, 19],
        [19, 25, 10],
    ]
    tensor_list = []
    for leg_link in leg_links:
        tensor_list.append(np.random.random_sample([10] * len(leg_link)))

    sequence = get_opt_sequence(tensor_list, leg_links, verbosity=True)
    print("Contraction Sequence:", sequence)
