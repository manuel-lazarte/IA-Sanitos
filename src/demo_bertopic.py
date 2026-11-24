"""Demostración real: python -m src.demo_bertopic [--csv ruta] [--offline]."""
import argparse
import json
from pathlib import Path

from src.bertopic_modelo import ConfigBERTopic
from src.carga_datos import cargar_csv
from src.pipeline import ejecutar_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="datos/noticias_demo.csv")
    parser.add_argument("--salida", default="output/bertopic/resumen.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    df, carga = cargar_csv(args.csv)
    resultado = ejecutar_pipeline(df, modelos=("BERTopic",),
                                 config_bertopic=ConfigBERTopic(local_files_only=args.offline))
    resumen = resultado["resumenes"]["BERTopic"]
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"carga": carga, "resumen": resumen,
                                  "noticias": df.to_dict(orient="records")},
                                 ensure_ascii=False, indent=2, default=str,
                                 allow_nan=False), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in (
        "n_documentos", "n_temas", "atipicos", "distribucion", "diversidad")},
        ensure_ascii=False, indent=2))
    print(f"Resultado completo: {destino}")


if __name__ == "__main__":
    main()
