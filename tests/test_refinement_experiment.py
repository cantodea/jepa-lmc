from __future__ import annotations

import json
import unittest

from experiments.cegar_refinement import PROTOCOL, assess


class RefinementScreenTests(unittest.TestCase):
    def test_exact_threshold_and_strict_shuffled_comparison(self) -> None:
        protocol = json.loads(PROTOCOL.read_text())

        def assessment(jepa: int, shuffled: int = 100) -> dict:
            methods = {
                name: {
                    "total_oracle_queries": queries,
                    "wrong_conclusive_verdicts": 0,
                    "unknown": 0,
                }
                for name, queries in (
                    ("cegar_jepa", jepa),
                    ("cegar_uniform", 100),
                    ("direct_bfs", 100),
                    ("cegar_shuffled", shuffled),
                )
            }
            return assess({42: methods}, protocol)

        self.assertEqual(
            assessment(80)["status"], "support_query_efficiency_followup"
        )
        self.assertEqual(
            assessment(81)["status"],
            "mechanism_valid_jepa_query_benefit_not_established",
        )
        self.assertEqual(
            assessment(80, shuffled=80)["status"],
            "mechanism_valid_jepa_query_benefit_not_established",
        )


if __name__ == "__main__":
    unittest.main()
