import argparse
from functools import partial

from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from optim.bayesian import BayesianOptimizer, cat_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = "../../input/siim-isic-melanoma-classification/"

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
    cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    objective = partial(
        cat_objective,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cat_cols=cat_cols,
        args=args.fold,
    )

    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.cat_save_params(study, args.params, cat_cols)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
