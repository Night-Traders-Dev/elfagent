from core.config import ROUTER_MODEL
from routing.policies import route_by_rules

class TaskRouter:
    def __init__(self):
        self.model_name = ROUTER_MODEL

    def route(self, query: str) -> dict:
        return route_by_rules(query)
