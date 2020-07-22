from monte_carlo import generate_input_data
from neural_network import inverse_planning_model

paths, goals = generate_input_data(100, 0.8)

model = inverse_planning_model()
# model.compile()
# model.fit(inputs, expected_outputs, batch_size, epochs)


def train():
    #treina a NN usando os dados  gerados por Monte Carlo
    pass


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    pass
