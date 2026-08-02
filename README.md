Introduction to Fermionic Time Evolution with Pytreenet (https://github.com/Drachier/PyTreeNet). 
To make the BUG integrator work please clone the Pytreenet repository and add the following change:

environment_block = contract_bra_tensor_ignore_one_leg(bra_tensor,bra_node,tensor,
                                                        op_node,
                                                        id_trafo_op(ignored_node_id), # ignored_node_id
                                                        id_trafo=id_trafo_bra)
                                        
This line can be found in state_operator_contraction.py around line 674.
This is the change specifically:

ignored_node_id -> id_trafo_op(ignored_node_id)