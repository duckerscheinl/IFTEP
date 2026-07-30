import numpy as np
import copy
from pytreenet.core import TreeTensorNetwork, Node
from pytreenet.random import crandn, random_tensor_node


def binary_tree_height(N):
    n = 1
    h = 0
    while N > n:
        n *= 2
        h += 1
    return h


def nodes_binary_tree(N, m, prefix, suffix, root="", parent_leg=0):

    ids = list()
    shapes = list()
    legs = list()
    atomic_ids = [[f"{prefix}{i}{suffix}", ""] for i in range(N)]
    atomic_shapes = [(2, 2) for i in range(N)]
    atomic_legs = np.empty(N, dtype=np.int16)
    atomic_legs[::2] = 0
    atomic_legs[1::2] = 1
    atomic_legs = list(atomic_legs)
    ids.append(atomic_ids)
    shapes.append(atomic_shapes)
    legs.append(atomic_legs)
    h = binary_tree_height(N=N)
    level = 0
    while level < h:
        prev_lvl_ids = ids[-1]
        prev_lvl_shs = shapes[-1]
        n_prev_lvl = len(prev_lvl_ids)
        next_lvl_ids = list()
        next_lvl_shs = list()
        next_lvl_legs = list()
        for i in range(0, n_prev_lvl - 1, 2):
            next_lvl_id = f"{prev_lvl_ids[i][0]}|{prev_lvl_ids[i+1][0]}"
            next_lvl_ids.append([next_lvl_id, ""])
            prev_lvl_ids[i][1] = next_lvl_id
            prev_lvl_ids[i + 1][1] = next_lvl_id
            d_l = prev_lvl_shs[i][0]
            d_r = prev_lvl_shs[i + 1][0]
            next_lvl_sh = (min(d_l * d_r, m), d_l, d_r, 1)
            next_lvl_shs.append(next_lvl_sh)
            next_lvl_legs.append((i // 2) % 2)

        if i != n_prev_lvl - 2:
            next_lvl_id = f"{prev_lvl_ids[i+2][0]}|"
            next_lvl_ids.append([next_lvl_id, ""])
            prev_lvl_ids[i + 2][1] = next_lvl_id
            next_lvl_sh = (prev_lvl_shs[i + 2][0], prev_lvl_shs[i + 2][0], 1)
            next_lvl_shs.append(next_lvl_sh)
            next_lvl_legs.append(((i + 2) // 2) % 2)
        ids.append(next_lvl_ids)
        shapes.append(next_lvl_shs)
        legs.append(next_lvl_legs)
        level += 1

    ids[-1][0][1] = root
    legs[-1][0] = parent_leg

    return ids, shapes, legs


def add_tree_to_ttn(
    ttn: TreeTensorNetwork, h: np.int32, ids: list, shapes: list, legs: list
):

    n_lvl = h + 1

    for j1 in range(n_lvl, 0, -1):
        j = j1 - 1
        lvl_ids = ids[j]
        lvl_shapes = shapes[j]
        lvl_legs = legs[j]
        for i, (node_id, parent_id) in enumerate(lvl_ids):
            tensor = crandn(lvl_shapes[i])
            leg = lvl_legs[i]
            node = Node(identifier=node_id)
            ttn.add_child_to_parent(
                child=node,
                tensor=tensor,
                child_leg=0,
                parent_id=parent_id,
                parent_leg=1 + leg,
            )


def random_impurity_star_ttn(Nb: np.int32, m: np.int32) -> TreeTensorNetwork:
    ttn = TreeTensorNetwork()

    meff = min(2 ** (Nb + 1), m)
    cnode_conn = [2 for i in range(Nb + 1)]
    iup_T_shape = copy.deepcopy(cnode_conn)
    iup_T_shape.insert(0, meff)
    iup_T = crandn(iup_T_shape)
    iup_ident = "ImpUp(root)"
    iup_node = Node(identifier=iup_ident)
    ttn.add_root(node=iup_node, tensor=iup_T)

    idown_T_shape = cnode_conn
    idown_T_shape.insert(0, meff)
    idown_T = crandn(idown_T_shape)
    idown_ident = "ImpDown"
    idown_node = Node(identifier=idown_ident)
    ttn.add_child_to_parent(
        child=idown_node, tensor=idown_T, child_leg=0, parent_id=iup_ident, parent_leg=0
    )

    for i in range(Nb):
        cup_ident = f"child{i}(up)"
        cup_shape = (2, 2)
        cup_node = random_tensor_node(shape=cup_shape, identifier=cup_ident)
        ttn.add_child_to_parent(
            child=cup_node[0],
            tensor=cup_node[1],
            child_leg=0,
            parent_id=iup_ident,
            parent_leg=i + 1,
        )
        cdown_ident = f"child{i}(down)"
        cdown_shape = (2, 2)
        cdown_node = random_tensor_node(shape=cdown_shape, identifier=cdown_ident)
        ttn.add_child_to_parent(
            child=cdown_node[0],
            tensor=cdown_node[1],
            child_leg=0,
            parent_id=idown_ident,
            parent_leg=i + 1,
        )

    return ttn


def random_impurity_binary_tree_ttn(Nb: np.int32, m: np.int32) -> TreeTensorNetwork:
    ttn = TreeTensorNetwork()

    h = binary_tree_height(N=Nb)
    up_ids, up_shapes, up_legs = nodes_binary_tree(
        N=Nb, m=m, prefix="child", suffix="(up)", root="ImpUp(root)"
    )
    down_ids, down_shapes, down_legs = nodes_binary_tree(
        N=Nb, m=m, prefix="child", suffix="(down)", root="ImpDown"
    )

    meff_I = min(2 ** (Nb + 1), m)
    meff_B = min(2**Nb, m)

    iup_T = crandn((meff_I, meff_B, 2))
    iup_ident = "ImpUp(root)"
    iup_node = Node(identifier=iup_ident)
    ttn.add_root(node=iup_node, tensor=iup_T)

    idown_T = crandn((meff_I, meff_B, 2))
    idown_ident = "ImpDown"
    idown_node = Node(identifier=idown_ident)
    ttn.add_child_to_parent(
        child=idown_node, tensor=idown_T, child_leg=0, parent_id=iup_ident, parent_leg=0
    )

    add_tree_to_ttn(ttn=ttn, h=h, ids=up_ids, shapes=up_shapes, legs=up_legs)
    add_tree_to_ttn(ttn=ttn, h=h, ids=down_ids, shapes=down_shapes, legs=down_legs)

    return ttn
