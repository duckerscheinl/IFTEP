import numpy as np
from pytreenet.core import TreeTensorNetwork, Node
from pytreenet.random import crandn


def binary_tree_height(N):
    n = 1
    h = 0
    while N > n:
        n *= 2
        h += 1
    return h


def nodes_binary_tree(n_leaves, chi_max, prefix, suffix, parent_id="", subtree_slot=0):

    """
    Leg 0 is the parent bond. Legs 1 + child_slot are child bonds.
    The final leg is physical (dimension 2 on bath sites and impurities, 1(dummy) on branch nodes).
    The root's leg 0 is a child bond, since it has no parent.
    """


    edges = list()
    shapes = list()
    slots = list()
    leaf_edges = [[f"{prefix}{i}{suffix}", ""] for i in range(n_leaves)]
    leaf_shapes = [(2, 2) for i in range(n_leaves)]
    leaf_slots = [i%2 for i in range(n_leaves)]
    edges.append(leaf_edges)
    shapes.append(leaf_shapes)
    slots.append(leaf_slots)

    h = binary_tree_height(N=n_leaves)
    level = 0
    while level < h:

        prev_lvl_edges = edges[-1]
        prev_lvl_shapes = shapes[-1]
        n_prev_lvl = len(prev_lvl_edges)
        next_lvl_edges = list()
        next_lvl_shapes = list()
        next_lvl_slots = list()

        for i in range(0, n_prev_lvl - 1, 2):
            next_lvl_id = f"{prev_lvl_edges[i][0]}|{prev_lvl_edges[i+1][0]}"
            next_lvl_edges.append([next_lvl_id, ""])
            prev_lvl_edges[i][1] = next_lvl_id
            prev_lvl_edges[i + 1][1] = next_lvl_id
            d_l = prev_lvl_shapes[i][0]
            d_r = prev_lvl_shapes[i + 1][0]
            next_lvl_shape = (min(d_l * d_r, chi_max), d_l, d_r, 1)
            next_lvl_shapes.append(next_lvl_shape)
            next_lvl_slots.append((i // 2) % 2)

        if i != n_prev_lvl - 2:
            next_lvl_id = f"{prev_lvl_edges[i+2][0]}|"
            next_lvl_edges.append([next_lvl_id, ""])
            prev_lvl_edges[i + 2][1] = next_lvl_id
            next_lvl_shape = (prev_lvl_shapes[i + 2][0], prev_lvl_shapes[i + 2][0], 1)
            next_lvl_shapes.append(next_lvl_shape)
            next_lvl_slots.append(((i + 2) // 2) % 2)

        edges.append(next_lvl_edges)
        shapes.append(next_lvl_shapes)
        slots.append(next_lvl_slots)
        level += 1

    edges[-1][0][1] = parent_id
    slots[-1][0] = subtree_slot

    return edges, shapes, slots


def add_tree_to_ttn(
    ttn: TreeTensorNetwork, h: np.int32, edges: list, shapes: list, slots: list
):

    n_lvl = h + 1

    for j1 in range(n_lvl, 0, -1):

        j = j1 - 1
        lvl_edges = edges[j]
        lvl_shapes = shapes[j]
        lvl_slots = slots[j]

        for i, (node_id, parent_id) in enumerate(lvl_edges):
            tensor = crandn(lvl_shapes[i])
            slot = lvl_slots[i]
            node = Node(identifier=node_id)
            ttn.add_child_to_parent(
                child=node,
                tensor=tensor,
                child_leg=0,
                parent_id=parent_id,
                parent_leg=1 + slot,
            )


def random_impurity_binary_tree_ttn(n_bath: np.int32, chi_max: np.int32) -> TreeTensorNetwork:
    ttn = TreeTensorNetwork()

    h = binary_tree_height(N=n_bath)
    up_edges, up_shapes, up_slots = nodes_binary_tree(
        n_leaves=n_bath, chi_max=chi_max, prefix="bath", suffix="(up)", parent_id="imp(up)"
    )
    down_edges, down_shapes, down_slots = nodes_binary_tree(
        n_leaves=n_bath, chi_max=chi_max, prefix="bath", suffix="(down)", parent_id="imp(down)"
    )

    chi_imp_side = min(2 ** (n_bath + 1), chi_max)
    chi_bath_side = min(2**n_bath, chi_max)

    iup_tensor = crandn((chi_imp_side, chi_bath_side, 2))
    iup_ident = "imp(up)"
    iup_node = Node(identifier=iup_ident)
    ttn.add_root(node=iup_node, tensor=iup_tensor)

    idown_tensor = crandn((chi_imp_side, chi_bath_side, 2))
    idown_ident = "imp(down)"
    idown_node = Node(identifier=idown_ident)
    ttn.add_child_to_parent(
        child=idown_node, tensor=idown_tensor, child_leg=0, parent_id=iup_ident, parent_leg=0
    )

    add_tree_to_ttn(ttn=ttn, h=h, edges=up_edges, shapes=up_shapes, slots=up_slots)
    add_tree_to_ttn(ttn=ttn, h=h, edges=down_edges, shapes=down_shapes, slots=down_slots)

    return ttn