#!/usr/bin/env python3

import pennylane as qp

dev = qp.device('default.qubit', wires=2)
#dev = qp.device('ionq.simulator', wires=2, noise_model='forte-enterprise-1')
#dev = qp.device('ionq.qpu', wires=2, backend='forte-enterprise-1')

@qp.set_shots(shots=100)
@qp.qnode(dev)
def circuit():
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.counts(wires=[0, 1])

result = circuit()
print(result)
