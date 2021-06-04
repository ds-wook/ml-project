import joblib
import optuna
from optuna.samplers import TPESampler


class BayesianOptimizer:
    def __init__(self, objective_function: object):
        self.objective_function = objective_function

    def build_study(self, trials: int, verbose: bool = False):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name="lgbm_parameter_opt",
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)
        return study

    def display_study_statistics(study: optuna.create_study):
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

    @staticmethod
    def save_params(study: optuna.create_study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["boosting_type"] = "gbdt"
        params["learning_rate"] = 0.05
        params["n_estimators"] = 10000
        params["objective"] = "binary"
        params["metric"] = "auc"
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def plot_optimization_history(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_optimization_history(study)

    @staticmethod
    def plot_param_importances(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_param_importances(study)

    @staticmethod
    def plot_edf(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_edf(study)
