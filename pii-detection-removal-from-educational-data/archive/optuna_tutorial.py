import optuna

def calc(hyparam):
    x = hyparam['x']
    return (x - 2) ** 2

def objective(trial):
    hyparam = {
        'x': trial.suggest_uniform('x', -10, 10)
    }
    res = calc(hyparam) 
    return res

def run_optuna():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_value = study.best_value
    print("best_params: ", best_params)
    print("best_value: ", best_value)

run_optuna()