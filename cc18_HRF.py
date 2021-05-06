import numpy as np
import openml
import pickle
import logging
import time
from tqdm import tqdm
import os

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from HistRF import HistRandomForestClassifier 
from BagDT import BaggedDecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

import argparse

def train_test(X, y, task_name, task_id, nominal_indices, args, clfs, save_path):
    # Set up Cross validation

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=0)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    if args.vary_samples:
        sample_sizes = np.logspace(
            np.log10(n_classes * 2),
            np.log10(np.floor(len(y)*(args.cv-1.1)/args.cv)),
            num=10,
            endpoint=True,
            dtype=int)        
    else:
        sample_sizes = [len(y)]

    # Check if existing experiments
    results_dict = {
        "task": task_name,
        "task_id": task_id,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "y": y,
        "test_indices": [],
        "n_estimators": args.n_estimators,
        "cv": args.cv,
        "nominal_features": len(nominal_indices),
        "sample_sizes": sample_sizes,
    }

    # Get numeric indices first
    numeric_indices = np.delete(np.arange(X.shape[1]), nominal_indices)

    # Numeric Preprocessing
    numeric_transformer = SimpleImputer(strategy="median")

    # Nominal preprocessing
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    transformers = []
    if len(numeric_indices) > 0:
        transformers += [("numeric", numeric_transformer, numeric_indices)]
    if len(nominal_indices) > 0:
        transformers += [("nominal", nominal_transformer, nominal_indices)]
    preprocessor = ColumnTransformer(transformers=transformers)

    _, n_features_fitted = preprocessor.fit_transform(X, y).shape
    results_dict["n_features_fitted"] = n_features_fitted
    print(f'Features={n_features}, nominal={len(nominal_indices)} (After transforming={n_features_fitted})')

    # Store training indices (random state insures consistent across clfs)
    for train_index, test_index in skf.split(X, y):
        results_dict["test_indices"].append(test_index)

    for clf_name, clf in clfs:
        pipeline = Pipeline(steps=[("Preprocessor", preprocessor), ("Estimator", clf)])

        fold_probas = []
        if not f"{clf_name}_metadata" in results_dict.keys():
            results_dict[f"{clf_name}_metadata"] = {}
        results_dict[f"{clf_name}_metadata"]["train_times"] = []
        results_dict[f"{clf_name}_metadata"]["test_times"] = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if args.vary_samples:
                stratified_sort = stratify_samplesizes(y_train, sample_sizes)
                X_train = X_train[stratified_sort]
                y_train = y_train[stratified_sort]

            probas_vs_sample_sizes = []

            for n_samples in sample_sizes:
                start_time = time.time()
                # Fix too few samples for internal CV of these methods
                if clf_name in ['IRF', 'SigRF'] and np.min(np.unique(y_train[:n_samples], return_counts=True)[1]) < 5:
                    print(f'{clf_name} requires more samples of minimum class. Skipping n={n_samples}')
                    y_proba = np.repeat(
                        np.bincount(y_train[:n_samples]).reshape(1, -1) / len(y_train[:n_samples]),
                        X_test.shape[0],
                        axis=0)
                    train_time = time.time() - start_time
                else:
                    pipeline = pipeline.fit(X_train[:n_samples], y_train[:n_samples])
                    train_time = time.time() - start_time
                    y_proba = pipeline.predict_proba(X_test)

                test_time = time.time() - (train_time + start_time)

                probas_vs_sample_sizes.append(y_proba)
                results_dict[f"{clf_name}_metadata"]["train_times"].append(train_time)
                results_dict[f"{clf_name}_metadata"]["test_times"].append(test_time)

            fold_probas.append(probas_vs_sample_sizes)

        results_dict[clf_name] = fold_probas
        print(f'{clf_name} Time: train_time={train_time:.3f}, test_time={test_time:.3f}, Cohen Kappa={cohen_kappa_score(y_test, y_proba.argmax(1)):.3f}, Accuracy={accuracy_score(y_test, y_proba.argmax(1)):.3f}')
    # If existing data, load and append to. Else save
    if os.path.isfile(save_path) and args.mode == 'OVERWRITE':
        logging.info(f"OVERWRITING {task_name} ({task_id})")
        with open(save_path, "rb") as f:
            prior_results = pickle.load(f)

        # Check these keys have the same values
        verify_keys = [
            "task",
            "task_id",
            "n_samples",
            "n_features",
            "n_classes",
            "y",
            "test_indices",
            "n_estimators",
            "cv",
            "nominal_features",
            "n_features_fitted",
            "sample_sizes"
        ]
        for key in verify_keys:
            assert _check_nested_equality(prior_results[key], results_dict[key]), \
                f'OVERWRITE {key} does not match saved value'

        # Replace/add data
        replace_keys = [name for name, _ in clfs]
        replace_keys += [f"{name}_metadata" for name in replace_keys]
        for key in replace_keys:
            prior_results[key] = results_dict[key]

        results_dict = prior_results

    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

def run_cc18(args, clfs):
    logging.basicConfig(
        filename="run_all.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )

    for key, val in vars(args).items():
        logging.info(f'{key}={val}')

    benchmark_suite = openml.study.get_suite(
        "OpenML-CC18"
    )  # obtain the benchmark suite

    folder = f"C:/Users/pteng/Desktop/NDD/HistRF/cc18_results/results_cv"
    if not os.path.exists(folder):
        os.makedirs(folder)

    def _run_task_helper(task_id):
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        task_name = task.get_dataset().name

        save_path = f"{folder}/{task_name}_results_dict.pkl"
        if args.mode == "OVERWRITE":
            if not os.path.isfile(save_path):
                logging.info(f"OVERWRITE MODE: Skipping {task_name} (doesn't  exist)")
                return
        elif args.mode == 'APPEND' and os.path.isfile(save_path):
            logging.info(f"APPEND MODE: Skipping {task_name} (already exists)")
            return

        print(f"{args.mode} {task_name} ({task_id})")
        logging.info(f"Running {task_name} ({task_id})")

        X, y = task.get_X_and_y()  # get the data

        nominal_indices = task.get_dataset().get_features_by_type(
            "nominal", [task.target_name]
        )
        try:
            train_test(X, y, task_name, task_id, nominal_indices, args, clfs, save_path)
        except Exception as e:
            print(f"Test {task_name} ({task_id}) Failed | X.shape={X.shape} | {len(nominal_indices)} nominal indices")
            print(e)
            logging.error(
                f"Test {task_name} ({task_id}) Failed | X.shape={X.shape} | {len(nominal_indices)} nominal indices"
            )
            import traceback
            logging.error(e)
            traceback.print_exc()

    task_ids_to_run = []
    for task_id in benchmark_suite.tasks:
        if args.start_id is not None and task_id < args.start_id:
            print(f'Skipping task_id={task_id}')
            logging.info(f'Skipping task_id={task_id}')
            continue
        if args.stop_id is not None and task_id >= args.stop_id:
            print(f'Stopping at task_id={task_id}')
            logging.info(f'Stopping at task_id={task_id}')
            break
        task_ids_to_run.append(task_id)

    if args.parallel_tasks is not None and args.parallel_tasks > 1:
        Parallel(n_jobs=args.parallel_tasks, verbose=1)(
            delayed(_run_task_helper)(
                    task_id
                    ) for task_id in tqdm(task_ids_to_run)
                )
    else:
        for task_id in tqdm(task_ids_to_run):  # iterate over all tasks
            _run_task_helper(task_id)
            
parser = argparse.ArgumentParser(description="Run CC18 dataset.")
parser.add_argument("--mode", action="store", default="CREATE", choices=["OVERWRITE", "CREATE", "APPEND"])
parser.add_argument("--cv", action="store", type=int, default=10)
parser.add_argument("--n_estimators", action="store", type=int, default=500)
parser.add_argument("--n_jobs", action="store", type=int, default=1)
parser.add_argument("--max_bins", action="store", type=int, default=255)
parser.add_argument("--max_features", action="store", default=None, help="Either an integer, float, or string in {'sqrt', 'log2'}. Default uses all features.")
parser.add_argument("--start_id", action="store", type=int, default=None)
parser.add_argument("--stop_id", action="store", type=int, default=None)
parser.add_argument("--honest_prior", action="store", default="ignore", choices=["ignore", "uniform", "empirical"])
parser.add_argument("--parallel_tasks", action="store", default=1, type=int)
parser.add_argument("--vary_samples", action="store_true", default=False)

args = parser.parse_args()
max_features = args.max_features
try:
    max_features = int(max_features)
except:
    try:
        max_features = float(max_features)
    except:
        pass

clfs = [
    (
        "HistRF",
        HistRandomForestClassifier(
            max_bins=255,
            n_estimators=args.n_estimators,
            verbose=1
        ),
    ),
    (
        "BagDT",
        BaggedDecisionTreeClassifier(
            n_estimators=args.n_estimators,
        )
    ),
    (
        "RF",
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_features=max_features,
            n_jobs=args.n_jobs),
    ),
]


def __main__():
    print(args)
    run_cc18(args, clfs)
    print("CC18")
    
__main__()