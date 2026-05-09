from qiskit.circuit.library import n_local, zz_feature_map
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

algorithm_globals.random_seed = 1376

training_features, training_labels, test_features, test_labels = ad_hoc_data(
    training_size=20, test_size=10, n=2, gap=0.3
)

feature_map = zz_feature_map(feature_dimension=2, reps=2, entanglement="linear")
ansatz = n_local(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=100))
vqc.fit(training_features, training_labels)

print(f"Testing accuracy: {vqc.score(test_features, test_labels):0.2f}")
