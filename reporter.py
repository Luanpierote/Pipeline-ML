from pipeline.runner import PipelineRunner

class Reporter:
    # Responsável por gerar relatórios
    def __init__(self, result: dict):
        runner = PipelineRunner()
        result = runner.run_pipeline()
        
        self.df = result["df_result"]
        self.summary = result["summary"]
        self.config = result["config"]
        
    # O método dataset_info gera um resumo do dataset
    def dataset_info(self):
        info = {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "missing_per_col": self.df.isnull().sum().to_dict(),
        }
        return info

    # O método model_metrics extrai métricas
    def model_metrics(self):
        return self.summary

    # O método display imprime um relatório legível
    def display(self):
        print("\n=== Dataset Info ===")
        info = self.dataset_info()
        print(f"Shape: {info['shape']}")
        print("Dtypes:")
        for col, dtype in info["dtypes"].items():
            print(f"  - {col}: {dtype}")
        print("Missing Values:")
        for col, n_missing in info["missing_per_col"].items():
            print(f"  - {col}: {n_missing}")

        print("\n=== Model Metrics ===")
        metrics = self.model_metrics()
        if not metrics:
            print("Nenhuma métrica de modelo disponível.")
        else:
            print("Métricas do modelo:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value}")
        