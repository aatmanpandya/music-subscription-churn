"""MLflow pyfunc wrapper so the registered model is loadable with plain
mlflow.pyfunc.load_model('models:/kkbox-churn@champion') or `mlflow models serve`."""

import mlflow.pyfunc


class ChurnPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from pipeline.scorer import ChurnScorer

        self.scorer = ChurnScorer.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return self.scorer.predict(model_input)
