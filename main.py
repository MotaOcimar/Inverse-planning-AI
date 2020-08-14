from monte_carlo import MonteCarlo
# from neural_network import inverse_planning_model


monte_carlo_instance = MonteCarlo()
nn_input, nn_output = monte_carlo_instance.generate_data(num_iterations=100, remaining=0.8)

# model = inverse_planning_model()
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(inputs, expected_outputs, batch_size, epochs)


def train():
    # treina a NN usando os dados  gerados por Monte Carlo
    pass


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    pass
