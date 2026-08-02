import numpy as np
import numpy.typing as npt
from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttno.state_diagram import TTNOFinder
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator, TreeTensorNetwork
from fractions import Fraction
from TreeTopologies import random_impurity_binary_tree_ttn


c = np.array([[0, 1], [0, 0]])
n = np.array([[0, 0], [0, 1]])
pz = np.array([[1, 0], [0, -1]])


def SIAM_TTNO(
    Nb: int,
    hop: npt.NDArray[np.complex128],
    eb: npt.NDArray[np.float64],
    ei: np.float64,
    U: np.float64,
    state: TreeTensorNetwork,
    eshift: float = 0,
    angle: float = 0,
    fact: Fraction = Fraction(1),
) -> TreeTensorNetworkOperator:

    hamiltonian = Hamiltonian()

    conversion_dict = {}
    conversion_dict["I2"] = np.eye(2)
    conversion_dict["I1"] = np.eye(1)
    conversion_dict["c"] = c
    conversion_dict["c*"] = c.T
    conversion_dict["n"] = n
    conversion_dict["pz"] = pz

    frac = fact
    coeff_map = dict()
    angle_fact = np.cos(angle) + 1j * np.sin(angle)
    angle_fact = np.real_if_close(angle_fact)
    angle_fact = -1j*(np.real_if_close(1j*(angle_fact)))

    e_shift_term = TensorProduct()
    e_shift_coeff_id = "E"
    coeff_map[e_shift_coeff_id] = eshift * angle_fact
    hamiltonian.add_term((frac, e_shift_coeff_id, e_shift_term))

    for i in range(Nb):

        pzs_up = {f"bath{j}(up)": "pz" for j in range(i + 1, Nb)}

        hop_term_up = TensorProduct(
            {f"bath{i}(up)": "c*"} | pzs_up | {"imp(up)": "c"}
        )
        hop_coeff_id = f"t{i}*"
        coeff_map[hop_coeff_id] = hop[i] * angle_fact
        hamiltonian.add_term((frac, hop_coeff_id, hop_term_up))
        hop_term_up_dag = TensorProduct(
            {f"bath{i}(up)": "c"} | pzs_up | {"imp(up)": "c*"}
        )
        hop_conj_coeff_id = f"t{i}"
        coeff_map[hop_conj_coeff_id] = np.conj(hop[i]) * angle_fact
        hamiltonian.add_term((frac, hop_conj_coeff_id, hop_term_up_dag))

        os_term_up = TensorProduct({f"bath{i}(up)": "n"})
        os_coeff_id = f"e{i}"
        coeff_map[os_coeff_id] = eb[i] * angle_fact
        hamiltonian.add_term((frac, os_coeff_id, os_term_up))

        pzs_down = {f"bath{j}(down)": "pz" for j in range(0, i)}
        hop_term_down = TensorProduct(
            {"imp(down)": "c"} | pzs_down | {f"bath{i}(down)": "c*"}
        )
        hamiltonian.add_term((frac, hop_coeff_id, hop_term_down))
        hop_term_down_dag = TensorProduct(
            {"imp(down)": "c*"} | pzs_down | {f"bath{i}(down)": "c"}
        )
        hamiltonian.add_term((frac, hop_conj_coeff_id, hop_term_down_dag))

        os_term_down = TensorProduct({f"bath{i}(down)": "n"})
        hamiltonian.add_term((frac, os_coeff_id, os_term_down))

    os_term_up = TensorProduct({"imp(up)": "n"})
    os_imp_coeff_id = "eimp"
    coeff_map[os_imp_coeff_id] = ei * angle_fact
    hamiltonian.add_term((frac, os_imp_coeff_id, os_term_up))
    os_term_down = TensorProduct({"imp(down)": "n"})
    hamiltonian.add_term((frac, os_imp_coeff_id, os_term_down))

    inter_term = TensorProduct({"imp(up)": "n", "imp(down)": "n"})
    inter_coeff_id = "U"
    coeff_map["U"] = U * angle_fact
    hamiltonian.add_term((frac, inter_coeff_id, inter_term))
    
    hamiltonian.conversion_dictionary = conversion_dict
    hamiltonian.coeffs_mapping = coeff_map

    ttno_ham = TreeTensorNetworkOperator.from_hamiltonian(
        hamiltonian=hamiltonian, reference_tree=state, method=TTNOFinder.SGE
    )

    return ttno_ham