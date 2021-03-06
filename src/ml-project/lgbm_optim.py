import argparse
from functools import partial

from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from optim.bayesian import BayesianOptimizer, lgbm_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    parse.add_argument(
        "--path", type=str, default="../../input/siim-isic-melanoma-classification/"
    )
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = args.path
    train = load_dataset(path)

    features = [
        "age_approx",
        "age_enc",
        "age_approx_mean_enc",
        "age_id_min",
        "age_id_max",
        "sex_enc",
        "anatom_enc",
        "n_images",
        "n_images_enc",
        "image_size_scaled",
        "image_size_enc",
    ]

    X = train[features]
    y = train["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    objective = partial(
        lgbm_objective, X_train=X_train, y_train=y_train, X_test=X_test, args=args.fold
    )

    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.lgbm_save_params(study, args.params)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
