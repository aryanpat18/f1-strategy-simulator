"""
dashboard/api_client.py
=======================

HTTP client used by the Streamlit dashboard to talk to the FastAPI backend.
"""

import requests
from typing import Dict, List, Optional


class F1StrategyAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # --------------------------------------------------
    # Simulation endpoints
    # --------------------------------------------------

    def auto_simulation(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/simulate/auto",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def manual_simulation(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/simulate/manual",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def optimize_strategy(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/simulate/optimize",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def get_degradation_curve(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/simulate/degradation",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def simulate_safety_car(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/simulate/safety-car",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def post_race_counterfactual(self, payload: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/post-race/counterfactual",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    # --------------------------------------------------
    # Data endpoints
    # --------------------------------------------------

    def list_races(self, year: Optional[int] = None) -> List[Dict]:
        params = {"year": year} if year is not None else {}
        response = requests.get(
            f"{self.base_url}/data/races",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def list_track_metrics(self) -> List[Dict]:
        response = requests.get(
            f"{self.base_url}/data/tracks",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def get_race_detail(self, year: int, round_number: int) -> Dict:
        response = requests.get(
            f"{self.base_url}/data/race/{year}/{round_number}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_race_analysis(self, year: int, round_number: int) -> Dict:
        response = requests.get(
            f"{self.base_url}/data/race-analysis/{year}/{round_number}",
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def get_pre_race_intelligence(self, year: int, event_name: str) -> Dict:
        response = requests.get(
            f"{self.base_url}/data/pre-race/{year}/{event_name}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def list_drivers(self, year: Optional[int] = None) -> List[str]:
        params = {"year": year} if year is not None else {}
        response = requests.get(
            f"{self.base_url}/data/drivers",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def list_teams(self, year: Optional[int] = None) -> List[str]:
        params = {"year": year} if year is not None else {}
        response = requests.get(
            f"{self.base_url}/data/teams",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict:
        response = requests.get(
            f"{self.base_url}/health",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
