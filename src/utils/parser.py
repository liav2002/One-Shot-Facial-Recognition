def parse_search_space(search_space):
    def parse_value(value):
        if isinstance(value, list):
            return tuple(parse_value(v) for v in value) if all(isinstance(v, list) for v in value) else [parse_value(v)
                                                                                                         for v in value]
        else:
            try:
                if isinstance(value, (float, int)) or 'e' in str(value) or '.' in str(value):
                    return float(value)
                elif str(value).isdigit():
                    return int(value)
                else:
                    return value
            except ValueError:
                return value

    return {key: parse_value(value) for key, value in search_space.items()}


def parse_hyperparameters(params):
    return {
        "batch_size": int(params["batch_size"]),
        "lr": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "betas": tuple(params["betas"]),
        "optimizer": str(params["optimizer"]),
        "loss_function": str(params["loss_function"])
    }
