from project import load_config, train_nn, generate_training_data

cfg = load_config()
x, y, t, T, sensor_data = generate_training_data(cfg)

result = train_nn(sensor_data, cfg)
