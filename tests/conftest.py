"""Fixtures compartidas para las pruebas del prototipo Grupo 15."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def ruta_demo() -> Path:
    """Ruta del corpus sintético de demostración (Task 2)."""
    return ROOT / "datos" / "noticias_demo.csv"


@pytest.fixture
def corpus_mini() -> list[str]:
    """8 textos crudos en español: 4 de combustibles y 4 de deportes.

    Corpus pequeño para pruebas rápidas de representación y modelos.
    """
    combustibles = [
        "El precio del diésel subió en Cochabamba y el transporte pidió "
        "abastecimiento de gasolina. Los conductores esperan combustible "
        "esta semana.",
        "La gasolina escasea en La Paz. El sector del transporte anuncia "
        "medidas por el abastecimiento de diésel en las estaciones.",
        "Productores del altiplano denuncian falta de diésel para la "
        "siembra. El combustible es clave para la maquinaria agrícola.",
        "El gobierno dispuso un plan de abastecimiento de gasolina y "
        "diésel. El precio del combustible se mantiene con subsidio.",
    ]
    deportes = [
        "El equipo ganó el partido en La Paz con dos goles del delantero. "
        "El torneo local sigue apasionante.",
        "El jugador anotó en el minuto final y su club clasificó al "
        "torneo internacional de fútbol.",
        "La selección entrenó de cara al partido por el torneo. El "
        "entrenador confirmó la lista de jugadores.",
        "Miles de hinchas celebraron el triunfo del equipo en el "
        "estadio. El próximo partido es por el campeonato.",
    ]
    return combustibles + deportes
