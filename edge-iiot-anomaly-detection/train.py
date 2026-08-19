"""Main execution orchestration script for training the ML Engine."""
import json
import logging
from datetime import datetime

from config.settings import config
from src.benchmarking.benchmark import PipelineBenchmarker
from src.data.loader import DataLoader
from src.data.profiler import DatasetProfiler
from src.data.splitter import DataSplitter
from src.explainability.shap_explainer import TreeShapExplainer
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector
from src.model.cross_validation import cross_validate_model
from src.model.evaluation import ModelEvaluator
from src.model.hyperparameter import tune_hyperparameters
from src.model.isolation_forest import AnomalyDetector
from src.model.threshold import ThresholdCalibrator
from src.preprocessing.pipeline import PreprocessingPipeline
from src.tracking.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Execute the full ML pipeline."""
    # Ensure directories exist
    config.paths.ensure_dirs()
    
    tracker = MLflowTracker()
    run_name = f"training_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    with tracker.start_run(run_name=run_name):
        tracker.log_params(config.model.__dict__)
        tracker.log_params(config.split.__dict__)
        
        # 1. Load Data
        logger.info("--- Step 1: Loading Data ---")
        dataset_path = config.paths.data_raw / config.dataset.filename
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run data download script first.")
            
        loader = DataLoader(dataset_path)
        df = loader.load()
        if not loader.validate(df):
            raise ValueError("Dataset validation failed.")
            
        # 2. Split Data
        logger.info("--- Step 2: Splitting Data ---")
        splitter = DataSplitter(
            train_ratio=config.split.train_ratio,
            val_ratio=config.split.val_ratio,
            test_ratio=config.split.test_ratio,
            random_state=config.split.random_state
        )
        split_result = splitter.split(
            df, 
            target_col=config.dataset.target_column,
            attack_type_col=config.dataset.attack_type_column
        )
        
        if not splitter.validate_split(split_result, len(df)):
            raise ValueError("Data split validation failed.")
            
        X_train, X_val, X_test = split_result.X_train, split_result.X_val, split_result.X_test
        y_val, y_test = split_result.y_val, split_result.y_test
        attack_types_val = split_result.attack_types_val
        attack_types_test = split_result.attack_types_test
        
        # 3. Preprocessing
        logger.info("--- Step 3: Preprocessing ---")
        pipeline = PreprocessingPipeline(
            columns_to_drop=config.dataset.columns_to_drop,
            target_columns=[config.dataset.attack_type_column, config.dataset.target_column]
        )
        
        X_train_processed = pipeline.fit_transform(X_train)
        X_val_processed = pipeline.transform(X_val)
        X_test_processed = pipeline.transform(X_test)
        
        # Save pipeline and feature columns
        pipeline_path = config.paths.artifacts_models / "preprocessing_pipeline.pkl"
        pipeline.save(pipeline_path)
        tracker.log_artifact(pipeline_path)
        
        feature_metadata = pipeline.get_feature_columns_json()
        with open(config.paths.artifacts_metadata / "feature_columns.json", "w") as f:
            json.dump(feature_metadata, f, indent=2)
        tracker.log_dict_as_json(feature_metadata, "feature_columns.json")
        
        # 4. Hyperparameter Tuning
        logger.info("--- Step 4: Hyperparameter Tuning ---")
        tuning_results = tune_hyperparameters(
            X_train_processed, X_val_processed, y_val, 
            n_trials=config.model.n_optuna_trials,
            random_state=config.random_seed
        )
        best_params = tuning_results["best_params"]
        logger.info(f"Best hyperparameters: {best_params}")
        tracker.log_params({f"best_{k}": v for k, v in best_params.items()})
        tracker.log_metrics({"best_val_f1_from_tuning": tuning_results["best_f1"]})
        
        # 5. Cross Validation
        logger.info("--- Step 5: Cross Validation ---")
        cv_results = cross_validate_model(
            X_train_processed, X_val_processed, y_val, 
            model_params=best_params,
            n_folds=config.model.cv_folds,
            random_state=config.random_seed
        )
        tracker.log_metrics({
            "cv_mean_f1": cv_results["mean_f1"],
            "cv_std_f1": cv_results["std_f1"],
            "cv_score_stability": cv_results["score_stability"]
        })
        
        # 6. Final Model Training & Threshold Calibration
        logger.info("--- Step 6: Final Model Training ---")
        final_model = AnomalyDetector(**best_params)
        final_model.fit(X_train_processed)
        
        # Threshold Calibration Comparison
        scores_train = final_model.score_samples(X_train_processed)
        scores_val = final_model.score_samples(X_val_processed)
        
        threshold_comp_df = ThresholdCalibrator.compare_all_methods(scores_train, scores_val, y_val.to_numpy())
        best_method, best_threshold = ThresholdCalibrator.select_best_threshold(
            threshold_comp_df, tie_epsilon=config.model.threshold_tie_epsilon
        )
        
        logger.info(f"Selected threshold method '{best_method}' with value: {best_threshold:.4f}")
        final_model.set_threshold(best_threshold)
        
        tracker.log_params({"selected_threshold_method": best_method, "threshold_value": best_threshold})
        
        # Save model
        model_path = config.paths.artifacts_models / "model.pkl"
        final_model.save(model_path)
        tracker.log_artifact(model_path)
        
        # 7. Final Evaluation on Test Set
        logger.info("--- Step 7: Final Evaluation on Test Set ---")
        scores_test = final_model.score_samples(X_test_processed)
        preds_test = final_model.predict(X_test_processed)
        
        evaluator = ModelEvaluator(config.paths.artifacts_plots)
        test_metrics = evaluator.compute_metrics(y_test.to_numpy(), preds_test, scores_test)
        
        logger.info(f"Test Set Metrics: {test_metrics}")
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        
        # Compute scores_normal and scores_attack for distribution plot
        scores_test_normal = scores_test[y_test == 0]
        scores_test_attack = scores_test[y_test == 1]
        
        # Generate plots
        plot_paths = evaluator.generate_all_plots(
            y_true=y_test.to_numpy(),
            y_pred=preds_test,
            scores=scores_test,
            attack_types=attack_types_test,
            threshold=best_threshold,
            comparison_df=threshold_comp_df,
            scores_normal=scores_test_normal,
            scores_attack=scores_test_attack
        )
        for p in plot_paths:
            tracker.log_artifact(p)
            
        # 8. SHAP Explainability
        logger.info("--- Step 8: SHAP Explainability ---")
        explainer = TreeShapExplainer(final_model.model, config.paths.artifacts_plots)
        
        # Use a subset of test data for SHAP to save time (max 1000 samples)
        shap_sample_size = min(1000, len(X_test_processed))
        X_shap = X_test_processed.sample(n=shap_sample_size, random_state=42)
        
        shap_values = explainer.compute_shap_values(X_shap)
        feature_names = pipeline.get_feature_names()
        
        shap_summary_path = explainer.plot_summary(shap_values, X_shap)
        shap_importance_path = explainer.plot_feature_importance(shap_values, X_shap)
        tracker.log_artifact(shap_summary_path)
        tracker.log_artifact(shap_importance_path)
        
        shap_json_path = explainer.export_global_importance(shap_values, feature_names)
        tracker.log_artifact(shap_json_path)
        
        # 9. Benchmarking
        logger.info("--- Step 9: Benchmarking ---")
        benchmarker = PipelineBenchmarker(pipeline, final_model, config.paths.artifacts_results)
        benchmark_results = benchmarker.run_benchmarks(X_test)
        tracker.log_dict_as_json(benchmark_results, "benchmark_results.json")
        
        # 10. Generate Metadata Artifact
        logger.info("--- Step 10: Generating Metadata ---")
        metadata = {
            "model_version": run_name,
            "schema_version": config.schema_version,
            "training_date": datetime.utcnow().isoformat() + "Z",
            "hyperparameters": best_params,
            "threshold": {
                "method": best_method,
                "value": float(best_threshold)
            },
            "test_metrics": test_metrics,
            "data_split_ratios": {
                "train": config.split.train_ratio,
                "val": config.split.val_ratio,
                "test": config.split.test_ratio
            }
        }
        
        metadata_path = config.paths.artifacts_metadata / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        tracker.log_artifact(metadata_path)
        
        # Generate final markdown report
        report_path = config.paths.artifacts_results / "results.md"
        with open(report_path, "w") as f:
            f.write(f"# Edge-IIoT Anomaly Detection - Training Results\n\n")
            f.write(f"**Run:** {run_name}\n")
            f.write(f"**Date:** {metadata['training_date']}\n\n")
            
            f.write(f"## Test Metrics\n")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    f.write(f"- **{k}:** {v:.4f}\n")
                elif not isinstance(v, dict):
                    f.write(f"- **{k}:** {v}\n")
                    
            f.write(f"\n## Selected Threshold\n")
            f.write(f"- **Method:** {best_method}\n")
            f.write(f"- **Value:** {best_threshold:.4f}\n")
            
        tracker.log_artifact(report_path)
        
        logger.info(f"Training pipeline complete! Results saved in {config.paths.artifacts_results}")

if __name__ == "__main__":
    main()
