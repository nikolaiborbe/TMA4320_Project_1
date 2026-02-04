from project import load_config, train_nn
cfg = load_config()

result = train_nn(sensor_data, cfg)

test_returns_tuple()