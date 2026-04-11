import unittest

from routing.heuristics import is_simple_code_request
from routing.policies import route_by_rules


class RoutingPolicyTests(unittest.TestCase):
    def test_weather_query_routes_to_web_research(self):
        route = route_by_rules("What is the current weather in Ashland, Kentucky?")
        self.assertEqual(route["route"], "web_research")

    def test_weather_query_is_not_simple_code(self):
        self.assertFalse(is_simple_code_request("What is the current weather in Ashland, Kentucky?"))

    def test_simple_script_request_still_uses_fast_code_heuristic(self):
        self.assertTrue(is_simple_code_request("Write a Python script that prints hello world."))


if __name__ == "__main__":
    unittest.main()
